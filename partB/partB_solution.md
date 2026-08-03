# Part B: Capacity reconcilation

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
