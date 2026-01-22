import timeit
import argparse
import torch
import csv
import os
from datetime import datetime
from cs336_basics.transformers_arch import TransformerLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--warmup_steps", type=int, default=5)
    args = parser.parse_args()

    model = TransformerLM(                                                                                                                                                                                                                                              
      vocab_size=args.vocab_size,
      context_length=args.context_length,
      d_model=args.d_model,
      num_layers=args.num_layers,
      num_heads=args.num_heads,
      d_ff=args.d_ff,
      rope_theta=10_000).to('cuda') # Mocking rope_theta

    for i in range(args.warmup_steps):
        input = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length))
        _ = model(input)

    input = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length))
    start_time_forward = timeit.default_timer()
    op = model(input)
    torch.cuda.synchronize()
    end_time_forward = timeit.default_timer()

    loss = op.mean()
    start_time_backward = timeit.default_timer()
    loss.backward()
    torch.cuda.synchronize()
    end_time_backward = timeit.default_timer()

    # Calculate elapsed times
    forward_time = end_time_forward - start_time_forward
    backward_time = end_time_backward - start_time_backward

    # Build result dict
    result = vars(args).copy()
    result['timestamp'] = datetime.now().isoformat()
    result['forward_time'] = forward_time
    result['backward_time'] = backward_time

    # Print to stdout
    print(result)

    # Write to CSV
    os.makedirs("benchmarks", exist_ok=True)
    csv_path = "benchmarks/python_timeit_benchmark.csv"
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

if __name__ == "__main__":
    main()