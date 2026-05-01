import math
import torch
import torch.nn.functional as F
import triton.testing
from itertools import product

from cs336_systems.flash_attention_2 import FlashAttentionPytorch
from cs336_systems.flash_attention_2_triton import FlashAttentionTriton

SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
D_MODELS = [16, 32, 64, 128]
BATCH_SIZE = 1
DTYPE = torch.bfloat16
DEVICE = "cuda"


def bench_forward(fn, q, k, v):
    return triton.testing.do_bench(lambda: fn(q, k, v, True))


def bench_backward(fn, q, k, v):
    do = torch.randn_like(q)

    def run():
        q.grad = k.grad = v.grad = None
        out = fn(q, k, v, True)
        out.backward(do)

    return triton.testing.do_bench(run)


def bench_end_to_end(fn, q, k, v):
    do = torch.randn_like(q)

    def run():
        q.grad = k.grad = v.grad = None
        out = fn(q, k, v, True)
        out.backward(do)

    return triton.testing.do_bench(run)


def sdpa_fn(q, k, v, is_causal):
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def make_inputs(seq_len, d_model):
    shape = (BATCH_SIZE, seq_len, d_model)
    q = torch.randn(shape, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k = torch.randn(shape, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(shape, device=DEVICE, dtype=DTYPE, requires_grad=True)
    return q, k, v


def run_benchmarks():
    rows = []

    for seq_len, d_model in product(SEQ_LENS, D_MODELS):
        print(f"Benchmarking seq_len={seq_len}, d_model={d_model}...")
        row = {"seq_len": seq_len, "d_model": d_model}

        impls = {
            "pytorch_sdpa": sdpa_fn,
            "fa2_pytorch": FlashAttentionPytorch.apply,
            "fa2_triton": FlashAttentionTriton.apply,
        }

        for name, fn in impls.items():
            try:
                q, k, v = make_inputs(seq_len, d_model)
                fwd_ms = bench_forward(fn, q, k, v)

                q, k, v = make_inputs(seq_len, d_model)
                bwd_ms = bench_backward(fn, q, k, v)

                q, k, v = make_inputs(seq_len, d_model)
                e2e_ms = bench_end_to_end(fn, q, k, v)

                row[f"{name}_fwd"] = f"{fwd_ms:.3f}"
                row[f"{name}_bwd"] = f"{bwd_ms:.3f}"
                row[f"{name}_e2e"] = f"{e2e_ms:.3f}"
            except Exception as e:
                row[f"{name}_fwd"] = "OOM"
                row[f"{name}_bwd"] = "OOM"
                row[f"{name}_e2e"] = "OOM"
                print(f"  {name} failed: {e}")

        rows.append(row)

    return rows


def rows_to_markdown(rows):
    headers = [
        "seq_len", "d_model",
        "sdpa_fwd", "sdpa_bwd", "sdpa_e2e",
        "fa2_pt_fwd", "fa2_pt_bwd", "fa2_pt_e2e",
        "fa2_tri_fwd", "fa2_tri_bwd", "fa2_tri_e2e",
    ]
    col_keys = [
        "seq_len", "d_model",
        "pytorch_sdpa_fwd", "pytorch_sdpa_bwd", "pytorch_sdpa_e2e",
        "fa2_pytorch_fwd", "fa2_pytorch_bwd", "fa2_pytorch_e2e",
        "fa2_triton_fwd", "fa2_triton_bwd", "fa2_triton_e2e",
    ]

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        cells = [str(row.get(k, "N/A")) for k in col_keys]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


if __name__ == "__main__":
    rows = run_benchmarks()
    md = rows_to_markdown(rows)

    out_path = "flash_benchmarking_results.md"
    with open(out_path, "w") as f:
        f.write("# Flash Attention Benchmarking Results\n\n")
        f.write("All times in milliseconds. Batch size 1, causal masking, dtype=bfloat16.\n\n")
        f.write(md)
        f.write("\n")

    print(f"\nResults written to {out_path}")
