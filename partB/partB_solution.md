# Part B: Capacity reconcilation

For the B1 calculations, the script `capacity.py` was used.

## B1 (a) Calculate exact KV-cache bytes per token
To calculate this we would use the bytes per token formula for tokenizers:
```
Bytes per token = 2 * KV_heads * head_dim * layers * Bytes_per_element
```
Where,

- KV_heads = Attention heads allocated for the KV pairs
- head_dim = The dimensions of 1 attention head
- layers = Total transformer layers used in model network
- bytes_per_element = the size used by per element as per the precision protocol.

According to model specs, the values are:

LAYERS = 28

KV_HEADS = 8 

HEAD_DIM = 128

BYTES_per_element = 2 (fp16 precision, 16bytes/8bits = 2)

therefore, 
Bytes per token = `2 * 8 * 128 * 28 * 2B` = `114,688 B` = `112 KB` (per token)


## B1 (b) Calculate  the approximate maximum number of concurrent 4096-token sequences this GPU can hold.

To calculate this we need the formula for maximum concurrent sequences that can run in memory:

```
Max Concurrency = floor(KV cache pool memory / Memory per sequence)
```
where

`KV cache pool memory = Memory (usable) - Memory weights - Memory overhead`

where 
`Memory weights = total parameters * bytes per element`

`Usable Memory = Total memory (VRAM) * Memory Utilization`


Our Values, after calculations:

`Usable Memory = 24GB * 0.92  = 22.08GB`

`Memory weights = 4.2e9 params * 2 B = 8.40 GB`

`overhead (given) = 1.60 GB`

`KV cache pool memory = 22.08GB - 8.40 - 1.60 = 12.08 GB`

`Memory per sequence = 114,688 B * 4096 = 0.4698 GB`

And finally, after plugging all values, we get:

Max concurrency  = `floor(12.08 / 0.4698) = floor(25.7)` 
 = 25 sequences

 ### After checking the prediction against the bench csv logs

Predicted KV utilization `= batch × (prompt+gen) × 114,688 B / KV budget`
matches the observed `kv_cache_util` **exactly** at every long-prompt row:

| batch | predicted util | observed `kv_cache_util` | `preempted_seqs` |
|---|---|---|---|
| 4 | 0.16 | 0.16 | 0 |
| 8 | 0.31 | 0.31 | 0 |
| 16 | 0.62 | 0.62 | 0 |
| 24 | 0.93 | 0.93 | 0 |
| 32 | **1.24** | 0.97 | 7 |
| 48 | **1.87** | 0.97 | 23 |

The prediction holds: preemption starts exactly when demanded KV exceeds the
budget (predicted util > 1.0), i.e. between batch 24 and batch 32. The capacity
model is validated by the log.


therefore Batch ~ 25 sequences


## B2 Long Context Sweep Anomaly

The anomaly is that previous approach expects reported tok/s to keep rising as batch size increases. But instead, it peaks at batch 24 (1607.4 tok/s) and falls back at batch 32 (1384.0) and batch 48 (1298.5)


Mechanism — KV-cache exhaustion → preemption → wasted recompute:

At batch 24, kv_cache_util = 0.93 — the cache is already nearly full (capacity is ~25 sequences, per B1).
At batch 32 the scheduler cannot hold all 32 sequences’ KV: preempted_seqs jumps 0 → 7, and at batch 48 → 23 (nearly half the batch). Utilization pins at 0.97.
A preempted sequence’s KV is evicted; when rescheduled, its prompt+generated prefix must be re-prefilled. That recompute consumes GPU time that produces no new output, so throughput drops instead of rising.
The latency columns confirm it: ttft_ms_p50 jumps 500 → 637 → 955 ms (requests wait/resubmit) and e2e_ms_p95 balloons 69 s → 97 s → 105 s, while itl_ms_p50 degrades only modestly (77 → 96 → 100 ms). The damage is in scheduling/recompute, not in steady-state decode speed.


Fix : enable fp8 KV cache. Halving KV bytes/token (114,688 → 57,344 B) doubles capacity to ~51 concurrent 4096-token sequences, so batch 32 and 48 fit with zero preemption. Predicted effect: preemptions 7/23 → 0; reported throughput at batch 32 returns to the pre-saturation trend (~1800–1900 tok/s vs the observed 1384; the batch-16→24 increment was +296 tok/s); e2e_ms_p95 back under ~75 s (vs 97.5 s). Cheaper alternative with the same mechanism: cap max_num_seqs at 24 and queue the excess — throughput then stays at the batch-24 level (~1600 reported) instead of degrading, at the cost of queuing latency for the overflow wave.


