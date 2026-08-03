# Part C — Decision memo: casual conversational style in 6 Indic languages

**Recommendation**: I think path (a) SFT on synthetic casualized pairs — with path (c ) built on day 1 as both the baseline and the data-generation engine. No need to build the rewriter (b).

## Assumptions

- The base model *can* produce casual register when prompted; the problem is
  default register, not capability. (Verified on day 1 — if false, this whole
  plan changes.)
- "Casual" is judgeable accept/reject by a native speaker in under a minute
  per response; deep rewriting is not required.
- Style transfer preserves content: we casualize the model's *own* answers, so
  factual quality is inherited, and the risk is limited to tone/naturalness.
- 3 weeks = 15 working days; A100 available for all of weeks 1–2.

## Back-of-envelope arithmetic

**Data volume.** Target 40k pairs (~6.5k × 6 languages): formal answer →
casual rewrite. Generation via the base model itself with a style prompt
(self-distillation; no external API needed). 40k pairs × ~400 output tokens ≈
16M tokens; a 4B model on one A100-80GB batches this at ~2–4k tok/s →
**~1.5–2.5 hours of generation**. Cheap.

**Reviewer throughput.** ~60–100 accept/reject judgments/hour → 10 h/week ≈
**700–1,000 items/week, ~2,500 over the project**, Hindi + Kannada only.
Plan: reviewer audits a random 10% sample of hin/kan pairs (~1,300 items) +
the full 200-item eval set per those languages. Tamil/Telugu/Bengali/Marathi
get automated filters (length ratio 0.6–1.4×, script purity, no-English-leak
except common loanwords) + base-model-as-judge, and are explicitly gated on
the hin/kan results (below). This is the plan's weakest point and we say so.

**Training cost.** LoRA SFT of the 4B model: 40k examples × ~800 tokens ≈
32M tokens × 2 epochs ≈ 64M tokens ≈ **12–20 h on one A100** — fits week 1
with days to spare. Full-parameter SFT of 4B (≈64 GB optimizer+weights in
fp16/bf16 Adam) also fits on 80 GB if LoRA under-delivers.

**Serving cost.** Zero incremental: style lives in the weights. Path (b) would
add a permanent second model per request — roughly +30–50% latency and a
second GPU-resident model eating the KV budget we just audited in Part B —
for a problem that is static (register doesn't change per request). That is
why (b) loses. Path ( c) alone stays as fallback: it costs ~100–200 extra
prompt tokens per request and is brittle across 6 languages, but it ships.

## Success metric (with threshold)

Blind pairwise A/B against current production outputs: reviewer prefers the
SFT model's casualness/naturalness in **≥ 70% of 200 held-out prompts** in
Hindi and Kannada, with **no regression** (≥ 95% parity) on a 100-item
content-correctness checklist. Tamil/Telugu/Bengali/Marathi ship only if the
hin/kan gate passes AND automated style markers (formality lexicon rate,
pronoun register) move by the same magnitude as in hin/kan.

## Kill criterion

**Go/no-go at end of week 2 (day 10).** Abandon SFT and ship the tuned prompt
( c) if either: (i) reviewer rejects **> 50%** of sampled synthetic pairs as
unnatural in Hindi or Kannada at the week-1 checkpoint (data engine broken);
or (ii) the day-10 A/B win rate is **< 55%** (model not learning the register
from this data). Both thresholds are observable a full week before launch
review, leaving week 3 to harden the prompt fallback.

## Day-1 experiment

Build the ( c) baseline and test the core assumption in one shot: write a
style prompt with 3–4 casual exemplars, generate answers to 100 real-ish
prompts in Hindi and Kannada, with and without the style prompt. Reviewer
blind-rates 50 pairs on day 2. This gives us (1) the baseline win rate every
later variant must beat, (2) proof (or disproof) that the base model can
produce the target register at all, and (3) the first ~100 seed pairs for the
SFT data engine. Cost: ~2 h of GPU, 1 h of reviewer time.
