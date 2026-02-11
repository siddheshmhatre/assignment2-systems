import timeit
import argparse
import torch
import csv
import os
import math
import torch.nn as nn
from datetime import datetime
from contextlib import nullcontext
import torch.cuda.nvtx as nvtx
from cs336_basics.transformers_training import AdamW
from einops import einsum

class AnnotatedScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = SoftMax()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        with nvtx.range("scaled_dot_product_attention"):
            d_k = Q.shape[-1]

            with nvtx.range("attention_scores"):
                attn_scores = einsum(Q, K, "... n d_k, ... m d_k -> ... n m")
                attn_scores = attn_scores / math.sqrt(d_k)
                if mask is not None:
                    attn_scores.masked_fill_(~mask, -math.inf)
            
            with nvtx.range("softmax"):
                attn_scores = self.softmax(attn_scores, dim=-1)

            with nvtx.range("final_matmul"):
                return einsum(attn_scores, V, "... n m, ... m d_v -> ... n d_v")

### Monkey patching ###
import cs336_basics.transformers_arch

cs336_basics.transformers_arch.ScaledDotProductAttention = AnnotatedScaledDotProductAttention

from cs336_basics.transformers_arch import TransformerLM, SoftMax
### Monkey patching ###

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--mem", action="store_true")
    parser.add_argument("--torch_compile", action="store_true")
    args = parser.parse_args()

    suffix = ""
    if args.amp:
        suffix += "_amp"
    if args.torch_compile:
        suffix += "_compile"

    amp_context_mgr = torch.autocast(device_type="cuda") if args.amp else nullcontext()

    model = TransformerLM(                                                                                                                                                                                                                                              
      vocab_size=args.vocab_size,
      context_length=args.context_length,
      d_model=args.d_model,
      num_layers=args.num_layers,
      num_heads=args.num_heads,
      d_ff=args.d_ff,
      rope_theta=10_000).to('cuda') # Mocking rope_theta

    if args.torch_compile:
        model = torch.compile(model)
    
    optim = AdamW(model.parameters())

    for i in range(args.warmup_steps):
        input = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length)).to('cuda')
        _ = model(input)

    input = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length)).to('cuda')

    if args.mem:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    with amp_context_mgr:
        start_time_forward = timeit.default_timer()
        with nvtx.range("forward pass"):
            op = model(input)
            torch.cuda.synchronize()
        end_time_forward = timeit.default_timer()

        if args.mem:
            os.makedirs("memory_snapshots", exist_ok=True)
            torch.cuda.memory._dump_snapshot(f"memory_snapshots/forward_pass_memory_snapshot_{args.context_length}{suffix}.pickle")

        loss = op.mean()
        start_time_backward = timeit.default_timer()
        with nvtx.range("backward pass"):
            loss.backward()
            torch.cuda.synchronize()
        end_time_backward = timeit.default_timer()

        start_time_optim = timeit.default_timer()
        with nvtx.range("optim step"):
            optim.step()
            torch.cuda.synchronize()
        end_time_optim = timeit.default_timer()

        if args.mem:
            torch.cuda.memory._dump_snapshot(f"memory_snapshots/training_step_{args.context_length}{suffix}.pickle")

        torch.cuda.memory._record_memory_history(enabled=None)

        model = model.eval()

        with nvtx.range("forward no grad"):
            with torch.no_grad():
                op = model(input)
            torch.cuda.synchronize()

    # Calculate elapsed times
    forward_time = end_time_forward - start_time_forward
    backward_time = end_time_backward - start_time_backward
    optim_time = end_time_optim - start_time_optim

    # Build result dict
    result = vars(args).copy()
    result['timestamp'] = datetime.now().isoformat()
    result['forward_time'] = forward_time
    result['backward_time'] = backward_time
    result['optim_time'] = optim_time

    # Print to stdout
    print(result)

    # Write to CSV
    os.makedirs("benchmarks", exist_ok=True)
    csv_path = f"benchmarks/python_timeit_benchmark{suffix}.csv"
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

if __name__ == "__main__":
    main()