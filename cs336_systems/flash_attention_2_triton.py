import math
import torch
import triton
import triton.language as tl

def cdiv(a, b):
    return (a + b - 1) // b

@triton.jit
def flash_attention_fwd(
	Q_ptr, K_ptr, V_ptr,
	O_ptr, L_ptr,
	stride_qb, stride_qq, stride_qd,
	stride_kb, stride_kk, stride_kd,
	stride_vb, stride_vk, stride_vd,
	stride_ob, stride_oq, stride_od,
	stride_lb, stride_lq,
	N_QUERIES, N_KEYS,
	scale,
	D: tl.constexpr,
	Q_TILE_SIZE: tl.constexpr,
	K_TILE_SIZE: tl.constexpr,
	is_causal: tl.constexpr,
):
	query_tile_index = tl.program_id(0)
	batch_index = tl.program_id(1)

	Q_block_ptr = tl.make_block_ptr(
		Q_ptr + batch_index * stride_qb,
		shape=(N_QUERIES, D),
		strides=(stride_qq, stride_qd),
		offsets=(query_tile_index * Q_TILE_SIZE, 0),
		block_shape=(Q_TILE_SIZE, D),
		order=(1,0),
	)

	O_block_ptr = tl.make_block_ptr(
		O_ptr + batch_index * stride_ob,
		shape=(N_QUERIES, D),
		strides=(stride_oq, stride_od),
		offsets=(query_tile_index * Q_TILE_SIZE, 0),
		block_shape=(Q_TILE_SIZE, D),
		order=(1, 0)
	)

	L_block_ptr = tl.make_block_ptr(
		L_ptr + batch_index * stride_lb,
		shape=(N_QUERIES, ),
		strides=(stride_lq, ),
		offsets=(query_tile_index * Q_TILE_SIZE, ),
		block_shape=(Q_TILE_SIZE, ),
		order=(0,),
	)

	K_block_ptr = tl.make_block_ptr(
		K_ptr + batch_index * stride_kb,
		shape=(N_KEYS, D),
		strides=(stride_kk, stride_kd),
		offsets=(0, 0),
		block_shape=(K_TILE_SIZE, D),
		order=(1, 0),
	)

	V_block_ptr = tl.make_block_ptr(
		V_ptr + batch_index * stride_vb,
		shape=(N_KEYS, D),
		strides=(stride_vk, stride_vd),
		offsets=(0, 0),
		block_shape=(K_TILE_SIZE, D),
		order=(1, 0),
	)

	output = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
	l = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
	m = tl.full((Q_TILE_SIZE,), -float('inf'), dtype=tl.float32)

	Q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")


	for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
		K_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
		V_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

		S = tl.dot(Q_block, tl.trans(K_block)) * scale

		if is_causal:
			q_indices = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
			k_indices = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
			mask = q_indices[:, None] < k_indices[None, :]
			S = tl.where(mask, S + (-1e6), S)

		old_m = m
		m = tl.maximum(m, tl.max(S, axis=1))
		P = tl.exp(S - m[:, None])

		correction = tl.exp(old_m - m)
		l = correction * l + tl.sum(P, axis=-1)
		output = tl.dot(P.to(V_ptr.type.element_ty), V_block, acc=correction[:, None] * output)

		K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
		V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

	output = ( 1 / l )[:, None] * output
	l = m + tl.log(l)

	output = output.to(O_ptr.type.element_ty)
	tl.store(O_block_ptr, output, boundary_check=(0, 1))
	tl.store(L_block_ptr, l, boundary_check=(0,))

@triton.jit
def flash_attention_bwd(
	Q_ptr, K_ptr, V_ptr,
	grad_O_ptr, D_ptr, L_ptr,
	grad_Q_ptr, grad_K_ptr, grad_V_ptr,
	stride_qb, stride_qq, stride_qd,
	stride_kb, stride_kk, stride_kd,
	stride_vb, stride_vk, stride_vd,
	stride_ob, stride_oq, stride_od,
	stride_db, stride_dq,
	stride_lb, stride_lq,
	N_QUERIES, N_KEYS,
	scale,
	D: tl.constexpr,
	Q_TILE_SIZE: tl.constexpr,
	K_TILE_SIZE: tl.constexpr,
	is_causal: tl.constexpr
):
	key_tile_index = tl.program_id(0)
	batch_index = tl.program_id(1)

	Q_block_ptr = tl.make_block_ptr(
		Q_ptr + batch_index * stride_qb,
		shape=(N_QUERIES, D),
		strides=(stride_qq, stride_qd),
		offsets=(0, 0),
		block_shape=(Q_TILE_SIZE, D),
		order=(1,0),
	)

	K_block_ptr = tl.make_block_ptr(
		K_ptr + batch_index * stride_kb,
		shape=(N_KEYS, D),
		strides=(stride_kk, stride_kd),
		offsets=(key_tile_index * K_TILE_SIZE, 0),
		block_shape=(K_TILE_SIZE, D),
		order=(1,0),
	)

	grad_K_block_ptr = tl.make_block_ptr(
		grad_K_ptr + batch_index * stride_kb,
		shape=(N_KEYS, D),
		strides=(stride_kk, stride_kd),
		offsets=(key_tile_index * K_TILE_SIZE, 0),
		block_shape=(K_TILE_SIZE, D),
		order=(1,0),
	)

	V_block_ptr = tl.make_block_ptr(
		V_ptr + batch_index * stride_vb,
		shape=(N_KEYS, D),
		strides=(stride_vk, stride_vd),
		offsets=(key_tile_index * K_TILE_SIZE, 0),
		block_shape=(K_TILE_SIZE, D),
		order=(1,0),
	)

	grad_V_block_ptr = tl.make_block_ptr(
		grad_V_ptr + batch_index * stride_vb,
		shape=(N_KEYS, D),
		strides=(stride_vk, stride_vd),
		offsets=(key_tile_index * K_TILE_SIZE, 0),
		block_shape=(K_TILE_SIZE, D),
		order=(1,0),
	)

	grad_O_block_ptr = tl.make_block_ptr(
		grad_O_ptr + batch_index * stride_ob,
		shape=(N_QUERIES, D),
		strides=(stride_oq, stride_od),
		offsets=(0, 0),
		block_shape=(Q_TILE_SIZE, D),
		order=(1,0),
	)

	D_block_ptr = tl.make_block_ptr(
		D_ptr + batch_index * stride_db,
		shape=(N_QUERIES, ),
		strides=(stride_dq, ),
		offsets=(0,),
		block_shape=(Q_TILE_SIZE,),
		order=(0, ),
	)

	L_block_ptr = tl.make_block_ptr(
		L_ptr + batch_index * stride_lb,
		shape=(N_QUERIES, ),
		strides=(stride_lq, ),
		offsets=(0,),
		block_shape=(Q_TILE_SIZE,),
		order=(0, ),
	)

	grad_K = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
	grad_V = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)

	K_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
	V_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

	for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
		Q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
		grad_O_block = tl.load(grad_O_block_ptr, boundary_check=(0, 1), padding_option="zero")
		L_block = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
		D_block = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")

		S = tl.dot(Q_block, tl.trans(K_block)) * scale

		if is_causal:
			q_indices = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
			k_indices = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
			causal_mask = q_indices[:, None] < k_indices[None, :]
			S = tl.where(causal_mask, S + (-1e6), S)

		P = tl.exp(S - L_block[:, None])

		P_cast = P.to(V_ptr.type.element_ty)
		grad_V = tl.dot(tl.trans(P_cast), grad_O_block, acc=grad_V)

		grad_P = tl.dot(grad_O_block, tl.trans(V_block))
		grad_S = P * (grad_P - D_block[:, None]) * scale

		q_offsets = i * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
		d_offsets = tl.arange(0, D)
		ptrs = grad_Q_ptr + batch_index * stride_qb + q_offsets[:, None] * stride_qq + d_offsets[None, :] * stride_qd
		q_mask = (q_offsets[:, None] < N_QUERIES) & (d_offsets[None, :] < D)
		grad_S_cast = grad_S.to(K_ptr.type.element_ty)
		tl.atomic_add(ptrs, tl.dot(grad_S_cast, K_block), mask=q_mask)

		grad_K = tl.dot(tl.trans(grad_S_cast), Q_block, acc=grad_K)

		Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
		grad_O_block_ptr = grad_O_block_ptr.advance((Q_TILE_SIZE, 0))
		L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
		D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

	tl.store(grad_K_block_ptr, grad_K, boundary_check=(0, 1))
	tl.store(grad_V_block_ptr, grad_V, boundary_check=(0, 1))

@torch.compile
def _flash_backward(Q, K, V, O, grad_O, L, is_causal):
	output_dims = Q.shape[-1]
	num_queries = Q.shape[1]
	num_keys = K.shape[1]
	scale = 1 / math.sqrt(output_dims)

	S = Q @ K.transpose(-1, -2) * scale # b, num_queries, num_keys

	if is_causal:
		Q_idxs = torch.arange(0, num_queries)
		K_idxs = torch.arange(0, num_keys)
		mask = (Q_idxs[:, None] < K_idxs[None, :]).to(Q.device)
		S = torch.where(mask[None, :], S - 1e6, S)
	P = torch.exp(S - L.unsqueeze_(-1)) # b, num_queries, num_keys
	dV = P.transpose(-1, -2) @ grad_O # b, keys, d
	dP = grad_O @ V.transpose(-1, -2) # b, num_queries, num_keys
	D = (O * grad_O).sum(dim=-1) # b, num_queries
	dS = P * (dP - D.unsqueeze_(-1)) # b, num_queries, num_keys
	dQ = dS @ K * scale # b, num_queries, d
	dK = dS.transpose(-1, -2) @ Q * scale # b, num_keys, d

	return dQ, dK, dV, None

def _flash_backward_triton(Q, K, V, O, grad_O, L, is_causal, Q_TILE_SIZE, K_TILE_SIZE):
	batch_size, num_queries, output_dims = Q.shape
	num_keys = K.shape[1]
	scale = 1 / math.sqrt(output_dims)

	grad_O = grad_O.contiguous()
	D = (O * grad_O).sum(dim=-1)  # (batch, num_queries)

	grad_Q = torch.zeros(Q.shape, device=Q.device, dtype=torch.float32)
	grad_K = torch.empty(K.shape, device=K.device, dtype=torch.float32)
	grad_V = torch.empty(V.shape, device=V.device, dtype=torch.float32)

	flash_attention_bwd[(cdiv(num_keys, K_TILE_SIZE), batch_size)](
		Q, K, V,
		grad_O, D, L,
		grad_Q, grad_K, grad_V,
		Q.stride(0), Q.stride(1), Q.stride(2),
		K.stride(0), K.stride(1), K.stride(2),
		V.stride(0), V.stride(1), V.stride(2),
		grad_O.stride(0), grad_O.stride(1), grad_O.stride(2),
		D.stride(0), D.stride(1),
		L.stride(0), L.stride(1),
		num_queries, num_keys,
		scale,
		output_dims,
		Q_TILE_SIZE,
		K_TILE_SIZE,
		is_causal,
	)

	return grad_Q.to(Q.dtype), grad_K.to(K.dtype), grad_V.to(V.dtype)

class FlashAttentionTriton(torch.autograd.Function):
	@staticmethod
	def forward(ctx, Q, K, V, is_causal=False, use_triton_bwd=False):
		# Q, K and V are of shape batch_size, seq_len, dim
		batch_size, num_queries, output_dims = Q.shape
		num_keys = K.shape[1]
		scale = 1 / math.sqrt(output_dims)
		ctx.output_dims = output_dims

		ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE = 32, 32
		ctx.is_causal = is_causal
		ctx.use_triton_bwd = use_triton_bwd

		O = torch.empty_like(Q)
		L = torch.empty((batch_size, num_queries), device=Q.device, dtype=torch.float32)

		flash_attention_fwd[(cdiv(num_queries, ctx.Q_TILE_SIZE), batch_size)](Q, K, V, O, L,
							num_queries * output_dims, output_dims, 1,
							num_keys * output_dims, output_dims, 1,
							num_keys * output_dims, output_dims, 1,
							num_queries * output_dims, output_dims, 1,
							num_queries, 1,
							num_queries, num_keys,
							scale,
							ctx.output_dims,
							ctx.Q_TILE_SIZE,
							ctx.K_TILE_SIZE,
							ctx.is_causal
					  )

		ctx.save_for_backward(Q, K, V, O, L)

		return O

	@staticmethod
	def backward(ctx, grad_O):
		Q, K, V, O, L = ctx.saved_tensors
		if ctx.use_triton_bwd:
			dQ, dK, dV = _flash_backward_triton(Q, K, V, O, grad_O, L, ctx.is_causal, ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE)
			return dQ, dK, dV, None, None
		else:
			dQ, dK, dV, _ = _flash_backward(Q, K, V, O, grad_O, L, ctx.is_causal)
			return dQ, dK, dV, None, None
