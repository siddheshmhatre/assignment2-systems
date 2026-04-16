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

	tl.store(O_block_ptr, output, boundary_check=(0, 1))
	tl.store(L_block_ptr, l, boundary_check=(0,))

class FlashAttentionTriton(torch.autograd.Function):
	@staticmethod
	def forward(ctx, Q, K, V, is_causal=False):
		# Q, K and V are of shape batch_size, seq_len, dim
		batch_size, num_queries, output_dims = Q.shape
		num_keys = K.shape[1]
		scale = 1 / math.sqrt(output_dims)
		ctx.output_dims = output_dims

		ctx.Q_TILE_SIZE, ctx.K_TILE_SIZE = 32, 32
		ctx.is_causal = is_causal

		O = torch.empty_like(Q)
		L = torch.empty((batch_size, num_queries), device=Q.device, dtype=Q.dtype)

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

		ctx.save_for_backward(Q, K, V, L)

		return O