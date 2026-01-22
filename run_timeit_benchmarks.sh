source .venv/bin/activate

echo "Running small model (12 layers, 12 heads, d_model=768)"
for i in {1..10}; do
    python python_timeit_benchmark.py --num_layers 12 --num_heads 12 --d_model 768 --d_ff 3072
	git add benchmarks/*
	git commit -m 'add file'
	git push
done

echo "Running medium model (24 layers, 16 heads, d_model=1024)"
for i in {1..10}; do
    python python_timeit_benchmark.py --num_layers 24 --num_heads 16 --d_model 1024 --d_ff 4096
done

echo "Running large model (36 layers, 20 heads, d_model=1280)"
for i in {1..10}; do
    python python_timeit_benchmark.py --num_layers 36 --num_heads 20 --d_model 1280 --d_ff 5120
done

echo "Running xl model (48 layers, 25 heads, d_model=1600)"
for i in {1..10}; do
    python python_timeit_benchmark.py --num_layers 48 --num_heads 25 --d_model 1600 --d_ff 6400
done

echo "Running 2.7B model (32 layers, 32 heads, d_model=2560)"
for i in {1..10}; do
    python python_timeit_benchmark.py --num_layers 32 --num_heads 32 --d_model 2560 --d_ff 10240
done

git add benchmarks/*
git commit -m 'add file'
git push