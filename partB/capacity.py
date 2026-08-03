
"""
capacity.py -- Part B: capacity reconciliation between model_spec.md
predictions and bench_log.csv observations. Every number in partB/partB_solution.md
is computed here.

"""

import csv
import sys

sys.stdout.reconfigure(encoding="utf-8")

# ---- model_spec.md
LAYERS = 28
KV_HEADS = 8          # GQA: KV heads, NOT the 24 Q heads
HEAD_DIM = 128
BYTES_FP16 = 2
PARAMS_B = 4.2e9
GPU_GB = 24.0
GPU_MEM_UTIL = 0.92
OVERHEAD_GB = 1.6
MAX_LEN = 4096

GB = 1e9  # spec units are decimal GB

print("=" * 72)
print("B1a: KV-cache bytes per token")
print("=" * 72)
kv_per_tok = LAYERS * 2 * KV_HEADS * HEAD_DIM * BYTES_FP16
print(f"layers*2(K,V)*kv_heads*head_dim*2B = 28*2*8*128*2 = {kv_per_tok:,} B"
      f" = {kv_per_tok / 1024:.0f} KiB/token")
print("(note: GQA makes this cheap, hence 8 KV heads and not 24 Q heads)")

print()
print("=" * 72)
print("B1b: max concurrent 4096-token sequences")
print("=" * 72)
usable = GPU_GB * GPU_MEM_UTIL
weights = PARAMS_B * BYTES_FP16 / GB
kv_budget = usable - weights - OVERHEAD_GB
per_seq = kv_per_tok * MAX_LEN / GB
max_seqs = kv_budget / per_seq
print(f"usable VRAM      = 24 GB * 0.92            = {usable:.2f} GB")
print(f"weights          = 4.2e9 params * 2 B      = {weights:.2f} GB")
print(f"overhead                                   = {OVERHEAD_GB:.2f} GB")
print(f"KV budget        = {usable:.2f} - {weights:.2f} - {OVERHEAD_GB:.2f}     = {kv_budget:.2f} GB")
print(f"per 4096-tok seq = {kv_per_tok:,} B * 4096   = {per_seq:.4f} GB")
print(f"max concurrent   = {kv_budget:.2f} / {per_seq:.4f}        = {max_seqs:.1f}"
      f"  -> floor = {int(max_seqs)} sequences")

print()
print("=" * 72)
print("B1c: check prediction against bench_log.csv")
print("=" * 72)
rows = list(csv.DictReader(open("../ASSIGNMENT/starter_kit/starter_kit/bench/bench_log.csv")))
print(f"{'batch':>6}{'prompt':>7}{'preempted':>10}{'kv_util':>9}{'pred_util':>10}")
for r in rows:
    b, p = int(r["batch_size"]), int(r["prompt_len"])
    if p < 3000:
        continue
    pred = b * (int(r["prompt_len"]) + int(r["gen_len"])) * kv_per_tok / GB / kv_budget
    print(f"{b:>6}{p:>7}{r['preempted_seqs']:>10}{r['kv_cache_util']:>9}{pred:>10.2f}")
print("prediction: preemption must start when predicted util > 1.0, i.e. batch > ~25")

print()
print("=" * 72)
print("B3: what reported_tok_s actually counts")
print("=" * 72)
print(f"{'batch':>6}{'prompt':>7}{'reported':>10}{'(p+g)*n/wall':>14}{'match?':>8}")
for r in rows:
    n, p, g = int(r["num_requests"]), int(r["prompt_len"]), int(r["gen_len"])
    wall = float(r["wall_clock_s"])
    recon = n * (p + g) / wall
    print(f"{n:>6}{p:>7}{float(r['reported_tok_s']):>10.1f}{recon:>14.1f}"
          f"{'YES' if abs(recon - float(r['reported_tok_s'])) < 0.5 else 'no':>8}")
print("-> reported_tok_s counts prompt (prefill) tokens as throughput.")

print()
print("=" * 72)
print("B3: honest goodput = generated tokens / second, two independent ways")
print("=" * 72)
print(f"{'batch':>6}{'prompt':>7}{'way1 n*g/wall':>15}{'way2 n*1000/itl':>17}")
for r in rows:
    n, p, g = int(r["num_requests"]), int(r["prompt_len"]), int(r["gen_len"])
    wall = float(r["wall_clock_s"])
    way1 = n * g / wall
    way2 = n * 1000.0 / float(r["itl_ms_p50"])
    print(f"{n:>6}{p:>7}{way1:>15.1f}{way2:>17.1f}")
print("way1 = end-to-end output goodput; way2 = decode-phase output rate from")
print("median ITL. Gap = prefill + scheduling time inside the wall clock.")

print()
print("=" * 72)
print("B2: throughput vs batch (prompt=3584) -- the anomaly")
print("=" * 72)
for r in rows:
    if int(r["prompt_len"]) == 3584:
        print(f"  batch {r['batch_size']:>2}: reported {float(r['reported_tok_s']):7.1f} tok/s, "
              f"goodput {int(r['num_requests']) * int(r['gen_len']) / float(r['wall_clock_s']):6.1f} tok/s, "
              f"preempted {r['preempted_seqs']:>2}, kv_util {r['kv_cache_util']}, "
              f"e2e_p95 {float(r['e2e_ms_p95']) / 1000:5.1f}s, ttft {r['ttft_ms_p50']}ms")
print("-> reported throughput FALLS past batch 24 exactly when preemption starts.")

print()
print("=" * 72)
print("B2 fix sizing: fp8 KV cache (halve bytes/token)")
print("=" * 72)
kv8 = kv_per_tok // 2
per_seq8 = kv8 * MAX_LEN / GB
print(f"fp8 KV: {kv8:,} B/token -> {per_seq8:.4f} GB/seq -> "
      f"{kv_budget / per_seq8:.1f} concurrent 4096-tok seqs (batch 48 fits)")
