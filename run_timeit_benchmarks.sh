for i in {1..10}; do
    python python_timeit_benchmark.py --num_layers 12 --num_heads 12 --d_model 768 --d_ff 3072
done

for i in {1..10}; do
    python python_timeit_benchmark.py --num_layers 24 --num_heads 16 --d_model 1024 --d_ff 4096
done

for i in {1..10}; do
    python python_timeit_benchmark.py --num_layers 36 --num_heads 20 --d_model 1280 --d_ff 5120
done

for i in {1..10}; do
    python python_timeit_benchmark.py --num_layers 48 --num_heads 25 --d_model 1600 --d_ff 6400
done

for i in {1..10}; do
    python python_timeit_benchmark.py --num_layers 32 --num_heads 32 --d_model 2560 --d_ff 10240
done