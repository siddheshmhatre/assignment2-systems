import torch
import timeit
import argparse
import csv
import os
from cs336_basics.transformers_arch import ScaledDotProductAttention


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_iters", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--torch_compile", action="store_true")
    args = parser.parse_args()

    d_model = args.d_model
    seq_len = args.seq_len

    print(f"d_model: {d_model}, seq_len: {seq_len}")

    Q, K, V = torch.randn(3, args.batch_size, seq_len, d_model, requires_grad=True).to("cuda")
    spda = ScaledDotProductAttention()

    if args.torch_compile:
        spda = torch.compile(spda)

    for k in range(args.warmup_steps):
        op = spda(Q, K, V)
        op.mean().backward()

    torch.cuda.memory._record_memory_history(max_entries=1000000)

    forward_pass_time = 0.0
    backward_pass_time = 0.0

    for i in range(args.num_iters):
        forward_pass_start_time = timeit.default_timer()
        op = spda(Q, K, V)
        torch.cuda.synchronize()
        forward_pass_end_time = timeit.default_timer()
        forward_pass_time += forward_pass_end_time - forward_pass_start_time

        if i == 0:
            if args.torch_compile:
                torch.cuda.memory._dump_snapshot(
                    f"memory_snapshots/forward_pass_memory_snapshot_d{d_model}_s{seq_len}_compile.pickle"
                )
            else:
                torch.cuda.memory._dump_snapshot(
                    f"memory_snapshots/forward_pass_memory_snapshot_d{d_model}_s{seq_len}.pickle"
                )

        backward_pass_start_time = timeit.default_timer()
        op.mean().backward()
        torch.cuda.synchronize()
        backward_pass_end_time = timeit.default_timer()
        backward_pass_time += backward_pass_end_time - backward_pass_start_time

        if i == 0:
            if args.torch_compile:
                torch.cuda.memory._dump_snapshot(
                    f"memory_snapshots/backward_pass_memory_snapshot_d{d_model}_s{seq_len}_compile.pickle"
                )
            else:
                torch.cuda.memory._dump_snapshot(
                    f"memory_snapshots/backward_pass_memory_snapshot_d{d_model}_s{seq_len}.pickle"
                )

    print(f"Total forward: {forward_pass_time:.6f}s, Total backward: {backward_pass_time:.6f}s")

    write_header = not os.path.exists(args.output_csv)
    with open(args.output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["d_model", "seq_len", "forward_time", "backward_time"])
        writer.writerow([d_model, seq_len, forward_pass_time, backward_pass_time])
