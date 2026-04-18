import math
import torch

@torch.compile
def _flash_backward(Q, K, V, O, grad_O, L):
	output_dims = Q.shape[-1]
	scale = 1 / math.sqrt(output_dims)

	S = Q @ K.transpose(-1, -2) * scale # b, num_queries, num_keys
	P = torch.exp(S - L.unsqueeze_(-1)) # b, num_queries, num_keys
	dV = P.transpose(-1, -2) @ grad_O # b, keys, d
	dP = grad_O @ V.transpose(-1, -2) # b, num_queries, num_keys
	D = (O * grad_O).sum(dim=-1) # b, num_queries
	dS = P * (dP - D.unsqueeze_(-1)) # b, num_queries, num_keys
	dQ = dS @ K * scale # b, num_queries, d
	dK = dS.transpose(-1, -2) @ Q * scale # b, num_keys, d

	return dQ, dK, dV, None

class FlashAttentionPytorch(torch.autograd.Function):
	@staticmethod
	def forward(ctx, Q, K, V, is_causal=False):
		tile_size = 16
		batch_size = Q.shape[0]
		seq_len = Q.shape[-2]
		dim = Q.shape[-1]

		O = torch.zeros_like(Q)
		L = torch.zeros(batch_size, seq_len).to(Q.device)
		num_iters = math.ceil(seq_len / tile_size)

		for i in range(num_iters):
			i_start = i * tile_size
			Q_i = Q[:, i_start: i_start + tile_size]             # (B, Ti, d)
			O_i = torch.zeros_like(Q_i)                           # (B, Ti, d)
			l_i = torch.zeros(batch_size, Q_i.shape[1]).to(Q.device)              # (B, Ti)
			m_i = torch.full((batch_size, Q_i.shape[1]), -float('inf')).to(Q.device)  # (B, Ti)

			for j in range(num_iters):
				j_start = j * tile_size
				K_j = K[:, j_start: j_start + tile_size]         # (B, Tj, d)
				V_j = V[:, j_start: j_start + tile_size]         # (B, Tj, d)

				S = (Q_i @ K_j.transpose(-2, -1)) / math.sqrt(dim)  # (B, Ti, Tj)
				old_mi = m_i                                          # (B, Ti)
				m_i = torch.max(m_i, torch.max(S, dim=-1).values)   # (B, Ti)
				P = torch.exp(S - m_i.unsqueeze(-1))                 # (B, Ti, Tj)

				correction = torch.exp(old_mi - m_i)                 # (B, Ti)
				l_i = correction * l_i + P.sum(dim=-1)               # (B, Ti)
				O_i = correction.unsqueeze(-1) * O_i + P @ V_j       # (B, Ti, d)

			O[:, i_start: i_start + tile_size] = (1 / l_i).unsqueeze(-1) * O_i
			L[:, i_start: i_start + tile_size] = m_i + torch.log(l_i)

		ctx.save_for_backward(Q, K, V, O, L)

		return O


	@staticmethod
	def backward(ctx, grad_O):
		Q, K, V, O, L = ctx.saved_tensors
		return _flash_backward(Q, K, V, O, grad_O, L)