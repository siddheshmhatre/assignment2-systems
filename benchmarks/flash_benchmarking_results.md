# Flash Attention Benchmarking Results

All times in milliseconds. Batch size 1, causal masking, dtype=bfloat16.

| seq_len | d_model | sdpa_fwd | sdpa_bwd | sdpa_e2e | fa2_pt_fwd | fa2_pt_bwd | fa2_pt_e2e | fa2_tri_fwd | fa2_tri_bwd | fa2_tri_e2e |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 128 | 16 | 0.056 | 0.099 | 0.093 | OOM | OOM | OOM | OOM | OOM | OOM |
| 128 | 32 | 0.056 | 0.095 | 0.094 | OOM | OOM | OOM | OOM | OOM | OOM |
| 128 | 64 | 0.055 | 0.095 | 0.096 | OOM | OOM | OOM | OOM | OOM | OOM |
| 128 | 128 | 0.061 | 0.108 | 0.103 | OOM | OOM | OOM | OOM | OOM | OOM |
| 256 | 16 | 0.061 | 0.106 | 0.270 | OOM | OOM | OOM | OOM | OOM | OOM |
| 256 | 32 | 0.060 | 0.124 | 0.104 | OOM | OOM | OOM | OOM | OOM | OOM |
| 256 | 64 | 0.060 | 0.105 | 0.103 | OOM | OOM | OOM | OOM | OOM | OOM |
| 256 | 128 | 0.060 | 0.112 | 0.111 | OOM | OOM | OOM | OOM | OOM | OOM |
| 512 | 16 | 0.066 | 0.163 | 0.117 | OOM | OOM | OOM | OOM | OOM | OOM |
| 512 | 32 | 0.065 | 0.117 | 0.117 | OOM | OOM | OOM | OOM | OOM | OOM |
| 512 | 64 | 0.070 | 0.125 | 0.125 | OOM | OOM | OOM | OOM | OOM | OOM |
| 512 | 128 | 0.078 | 0.167 | 0.155 | OOM | OOM | OOM | OOM | OOM | OOM |
| 1024 | 16 | 0.114 | 0.203 | 0.194 | OOM | OOM | OOM | OOM | OOM | OOM |
| 1024 | 32 | 0.117 | 0.209 | 0.211 | OOM | OOM | OOM | OOM | OOM | OOM |
| 1024 | 64 | 0.128 | 0.267 | 0.252 | OOM | OOM | OOM | OOM | OOM | OOM |
| 1024 | 128 | 0.149 | 0.295 | 0.293 | OOM | OOM | OOM | OOM | OOM | OOM |
| 2048 | 16 | 0.404 | 0.717 | 0.689 | OOM | OOM | OOM | OOM | OOM | OOM |
| 2048 | 32 | 0.389 | 0.693 | 0.674 | OOM | OOM | OOM | OOM | OOM | OOM |
| 2048 | 64 | 0.426 | 0.772 | 0.785 | OOM | OOM | OOM | OOM | OOM | OOM |
| 2048 | 128 | 0.505 | 0.992 | 0.981 | OOM | OOM | OOM | OOM | OOM | OOM |
| 4096 | 16 | 1.380 | 2.315 | 2.278 | OOM | OOM | OOM | OOM | OOM | OOM |
| 4096 | 32 | 1.314 | 2.399 | 2.550 | OOM | OOM | OOM | OOM | OOM | OOM |
| 4096 | 64 | 1.451 | 2.576 | 2.600 | OOM | OOM | OOM | OOM | OOM | OOM |
| 4096 | 128 | 1.621 | 3.032 | 3.006 | OOM | OOM | OOM | OOM | OOM | OOM |
| 8192 | 16 | 5.337 | 9.150 | 9.212 | OOM | OOM | OOM | OOM | OOM | OOM |
| 8192 | 32 | 5.239 | 9.313 | 9.231 | OOM | OOM | OOM | OOM | OOM | OOM |
| 8192 | 64 | 5.533 | 9.734 | 10.024 | OOM | OOM | OOM | OOM | OOM | OOM |
| 8192 | 128 | 5.954 | 11.583 | 11.387 | OOM | OOM | OOM | OOM | OOM | OOM |
| 16384 | 16 | 20.635 | 36.988 | 35.743 | OOM | OOM | OOM | OOM | OOM | OOM |
| 16384 | 32 | 21.774 | 35.903 | 35.636 | OOM | OOM | OOM | OOM | OOM | OOM |
| 16384 | 64 | 21.418 | 36.929 | 36.921 | OOM | OOM | OOM | OOM | OOM | OOM |
| 16384 | 128 | 23.051 | 43.127 | 43.138 | OOM | OOM | OOM | OOM | OOM | OOM |
| 32768 | 16 | 90.649 | 145.608 | 147.233 | OOM | OOM | OOM | OOM | OOM | OOM |
| 32768 | 32 | 89.483 | 148.889 | 154.240 | OOM | OOM | OOM | OOM | OOM | OOM |
| 32768 | 64 | 90.326 | 154.426 | 154.138 | OOM | OOM | OOM | OOM | OOM | OOM |
| 32768 | 128 | 107.358 | 181.931 | 187.036 | OOM | OOM | OOM | OOM | OOM | OOM |
| 65536 | 16 | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM |
| 65536 | 32 | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM |
| 65536 | 64 | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM |
| 65536 | 128 | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM | OOM |
