#!/bin/bash

EXTRA_ARGS="$@"

if [[ " $EXTRA_ARGS " == *" --torch_compile "* ]]; then
  CSV="benchmarks/attention_benchmark_compile.csv"
else
  CSV="benchmarks/attention_benchmark.csv"
fi
rm -f "$CSV"

D_MODELS=(16 32 64 128)
SEQ_LENS=(256 1024 4096 8192 16384)

for d in "${D_MODELS[@]}"; do
  for s in "${SEQ_LENS[@]}"; do
    echo "Running d_model=$d seq_len=$s"
    python benchmark_attention.py --d_model "$d" --seq_len "$s" --output_csv "$CSV" $EXTRA_ARGS
  done
done

echo "Results written to $CSV"
