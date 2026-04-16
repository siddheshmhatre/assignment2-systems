import timeit
import torch
import triton
import triton.language as tl
from einops import rearrange

def cdiv(a, b):
    return (a + b - 1) // b

@triton.jit # Why do need this?
def weighted_sum_fwd(
    x_ptr, weight_ptr,
    output_ptr,
    x_stride_row, x_stride_dim,
    weight_stride_dim,
    output_stride_row,
    ROWS, D,
    ROW_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr
):
    row_tile_idx = tl.program_id(0)

    x_block_ptr = tl.make_block_ptr(x_ptr, 
                                 (ROWS, D),
                                 (x_stride_row, x_stride_dim),
                                 (row_tile_idx * ROW_TILE_SIZE, 0),
                                 (ROW_TILE_SIZE, D_TILE_SIZE),
                                 (1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(weight_ptr,
                                      (D,),
                                      (weight_stride_dim,),
                                      (0,),
                                      (D_TILE_SIZE,),
                                      (0, ),
    )
    output_block_ptr = tl.make_block_ptr(output_ptr,
                                      (ROWS,),
                                      (output_stride_row,),
                                      (row_tile_idx * ROW_TILE_SIZE),
                                      (ROW_TILE_SIZE,),
                                      (0, ),
                                      )

    output = tl.zeros((ROW_TILE_SIZE, ), dtype=tl.float32)

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") #TODO - remove and try
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")

        output += tl.sum(x * weight[None, :], axis=1)

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE, ))

    tl.store(output_block_ptr, output, boundary_check=(0,))

@triton.jit
def weighted_sum_bwd(
    x_ptr, weight_ptr,
    grad_output_ptr,
    grad_x_ptr, partial_grad_weight_ptr,
    x_stride_row, x_stride_dim,
    weight_stride_dim,
    grad_output_ptr_stride_row,
    grad_x_ptr_stride_row, grad_x_ptr_stride_dim,
    partial_grad_weight_ptr_stride_row, partial_grad_weight_ptr_stride_dim,
    ROWS, D,
    ROW_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr
):
    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0) # Not sure where this is coming from
    x_block_ptr = tl.make_block_ptr(x_ptr, 
                                 (ROWS, D),
                                 (x_stride_row, x_stride_dim),
                                 (row_tile_idx * ROW_TILE_SIZE, 0),
                                 (ROW_TILE_SIZE, D_TILE_SIZE),
                                 (1, 0),
    )
    weight_block_ptr = tl.make_block_ptr(weight_ptr,
                                      (D,),
                                      (weight_stride_dim,),
                                      (0,),
                                      (D_TILE_SIZE,),
                                      (0, ),
    )
    grad_block_ptr = tl.make_block_ptr(grad_output_ptr,
                                      (ROWS,),
                                      (grad_output_ptr_stride_row,),
                                      (row_tile_idx * ROW_TILE_SIZE,),
                                      (ROW_TILE_SIZE,),
                                      (0, ),
    )
    grad_x_block_ptr = tl.make_block_ptr(grad_x_ptr,
                                        (ROWS, D),
                                        (grad_x_ptr_stride_row, grad_x_ptr_stride_dim),
                                        (row_tile_idx * ROW_TILE_SIZE, 0),
                                        (ROW_TILE_SIZE, D_TILE_SIZE),
                                        (1, 0),
    )
    partial_grad_weight_block_ptr = tl.make_block_ptr(partial_grad_weight_ptr,
                                                     (n_row_tiles, D),
                                                     (partial_grad_weight_ptr_stride_row, partial_grad_weight_ptr_stride_dim),
                                                     (row_tile_idx, 0),
                                                     (1, D_TILE_SIZE),
                                                     (1, 0),
    )

    grad_output = tl.load(grad_block_ptr, boundary_check=(0,), padding_option="zero")
    for i in range (tl.cdiv(D, D_TILE_SIZE)):
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")

        grad_x = grad_output[:, None] * weight[None, :]
        tl.store(grad_x_block_ptr, grad_x, boundary_check=(0, 1))

        x = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")
        
        partial_grad_weight = tl.sum(grad_output[:, None] * x, axis=0, keep_dims=True) # Why do we need keep_dims
        tl.store(partial_grad_weight_block_ptr, partial_grad_weight, boundary_check=(1, )) # Never out of bounds for dim 0 ( Is this a consequence of keep_dims?)

        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))

class WeightedSumFunc(torch.autograd.Function):
    @staticmethod # Why does it need to be a staticmethod
    def forward(ctx, x, weight):
        D, output_dims = x.shape[-1], x.shape[:-1]
        input_shape = x.shape
        
        x = rearrange(x, "... d -> (...) d")
        ctx.save_for_backward(x, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.D_TILE_SIZE = max(16, triton.next_power_of_2(D) // 16)
        ctx.ROWS_TILE_SIZE = 16 
        ctx.input_shape = input_shape

        n_rows = x.shape[0]
        
        y = torch.empty(n_rows, device=x.device)


        weighted_sum_fwd[(cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight,
            y,
            x.stride(0), x.stride(1),
            weight.stride(0),
            y.stride(0),
            ROWS=n_rows, D=D,
            ROW_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE,
        )

        return y.view(output_dims)

    @staticmethod
    def backward(ctx, grad_out):
        # Retrieve saved tensors and tile sizes from context
        x, weight = ctx.saved_tensors
        ROW_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE
        n_rows, D = x.shape

        # Reshape grad_out to be 1D to match the flattened rows of x
        grad_out = grad_out.reshape(-1)

        # Initialize output buffers
        # partial_grad_weight stores the weight gradient contribution from each thread block
        partial_grad_weight = torch.empty((cdiv(n_rows, ROW_TILE_SIZE), D), device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)

        # Launch the backward kernel
        weighted_sum_bwd[(cdiv(n_rows, ROW_TILE_SIZE),)](
            x, weight,
            grad_out,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1),
            weight.stride(0),
            grad_out.stride(0),
            grad_x.stride(0), grad_x.stride(1),
            partial_grad_weight.stride(0), partial_grad_weight.stride(1),
            ROWS=n_rows, D=D,
            ROW_TILE_SIZE=ROW_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
        )

        # Reduce the partial weight gradients along the rows to get the final gradient vector
        grad_weight = partial_grad_weight.sum(axis=0)

        # Return the gradients for x and weight (matching the order of forward inputs)
        return grad_x.view(ctx.input_shape), grad_weight

def weighted_sum(x, weight):
    return (weight * x).sum(axis=-1)

if __name__ == "__main__":
    batch_size, seq_len, embed_dim = 4, 128, 16384
    
    x = torch.randn((batch_size, seq_len, embed_dim), device='cuda', requires_grad=True)
    weight = torch.randn((embed_dim,), device='cuda', requires_grad=True)

    # Do warmup -
    for _ in range(5):
        output = WeightedSumFunc.apply(x, weight)
        output.sum().backward()
    start_time = timeit.default_timer()
    output_triton = WeightedSumFunc.apply(x, weight)
    end_time = timeit.default_timer()
    triton_time = end_time - start_time
    print(f"Output shape: {output_triton.shape}, Triton time: {triton_time}")

    start_time = timeit.default_timer()
    output_triton.sum().backward()
    end_time = timeit.default_timer()
    triton_bwd_time = end_time - start_time
    grad_x_triton, grad_w_triton = x.grad, weight.grad
    print(f"Grad_x shape: {grad_x_triton.shape}, Grad_w shape: {grad_w_triton.shape}, Triton time: {triton_bwd_time}")

    start_time = timeit.default_timer()
    output = weighted_sum(x, weight)
    end_time = timeit.default_timer()
    pytorch_time = end_time - start_time
    print(f"Output shape: {output.shape}, Triton time: {pytorch_time}")

    start_time = timeit.default_timer()
    output.sum().backward()
    end_time = timeit.default_timer()
    pytorch_bwd_time = end_time - start_time
    grad_x, grad_w = x.grad, weight.grad
    print(f"Grad_x shape: {grad_x.shape}, Grad_w shape: {grad_w.shape}, Triton time: {pytorch_bwd_time}")

    print (f"Pytorch output == Triton output {torch.allclose(output, output_triton, atol=1e-5, rtol=1e-4)}")
    print (f"Pytorch grad_x == Triton grad_x {torch.allclose(grad_x, grad_x_triton, atol=1e-5, rtol=1e-4)}")
    print (f"Pytorch grad_w == Triton grad_w {torch.allclose(grad_w, grad_w_triton, atol=1e-5, rtol=1e-4)}")


    print (f"Speedup - {pytorch_time / triton_time}")
    print (f"Speedup bwd - {pytorch_bwd_time / triton_bwd_time}")