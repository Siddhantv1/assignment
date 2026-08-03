# NOTEBOOK.md Lab Notes

Auditing `REPORT_v0.md` (tokenizer fertility + serving capacity).

Working directory: `ASSIGNMENT/`.

Environment: Python 3.13 venv called `myvenv` with `tiktoken`, `transformers` (tokenizers only), and `regex`.

3rd Aug, 2026 6pm

---

## #0  How would I recreate/reproduce this report

To run it manually, I used the usual python commands.

```bash
cd starter_kit

python fertility.py \
  --corpus eng=corpus_sample/eng_sample.txt \
  --corpus hin=corpus_sample/hin_sample.txt \
  --tokenizer gpt2
```

Results :

- eng: 1.27 / 0.226
- hin: 7.45 / 1.579
- ratio: 5.89×

Yup, matches `REPORT_v0.md` exactly, square one it is

---

## #1 Finding a better corpus

Wanted a parallel corpus so language comparisons would actually be fair.

Ask Gemini about the FLORES-200 Corpus:

- old FLORES GitHub path → 404
- guessed `tar.bz2` download URL → 403
- Hugging Face mirrors → gated or only dataset scripts

Eventually had to look at the loading script in `Muennighoff/flores200` on HuggingFace, where I found the valid archive:

`https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz`

Downloaded (~25 MB), extracted the five dev files (997 sentences each), and spot-checked that the lines really were parallel.

Note: A lot of blog posts still point at outdated FLORES locations. AI had difficulty navigating the web search.

---

## #2 Double-space bug (E1)

The code uses `line.split(" ")`, which treats repeated spaces as empty words. Why?

The sample corpus actually has a couple of double spaces (eng line 7, hin line 10), so I asked AI for quick script (`bug_exp.py`, E1).

On the sample corpus:

- English fertility shifts by about 1.3%
- Hindi by about 1.6%

This HAS to be a bug, in production it will matter at a lot larger scale.

---

## #3 Lowercasing (E2)

I expected lowercasing to reduce GPT-2 token counts in English. But no.

On FLORES English:

- original: 25,741 tokens
- lowercased: 26,696 tokens (+3.71%)

GPT-2 has merges that favor common sentence with capital letter starts (e.g. " The").

Hindi is basically unchanged (+0.01%).

So `lower()` is in reality shrinking the reported Hindi/English gap instead of enlarging it. 
Was good to measure before writing that section.

---

## #4 Mean of ratios vs aggregate (E3)

Measured the effect of averaging per-line fertility instead of computing it over the whole corpus.

Difference is tiny on FLORES:

- +0.7% English
- +0.4% Hindi

Still, aggregate is the right metric because token cost is additive. Short lines shouldn't have the same weight as long ones.

---

## #5 NFC normalization (E4)

I thought this would be a complete no-op, lol no.

About 90 of the 997 Hindi lines aren't NFC, and normalization changes GPT-2 token count by roughly 0.12% (191,589 → 191,828).

Also tried a deliberately ugly NFD example and GPT-2 barely cared.

Therfore: keep normalization. The effect is tiny, but it's doing the right thing, doesn't harm enough to be called a bug.

---

## #6 Tokenizer choice (E5)

haha This is big.

Ran the same FLORES corpus through GPT-2 and XLM-R.

Hindi goes from:

- 192.4 tokens/sentence (GPT-2)
- 36.7 tokens/sentence (XLM-R)

The Hindi/English ratio drops from 7.45× to about 1.26×.

Kannada, Tamil, and Telugu show the same pattern.

That clears it that old report's gap is not caused by the script. Most of it comes from using an English tokenizer.

---

## #7 Metric choice (E6/E7)

Even with the calculations fixed, "tokens per whitespace word" doesn't really compare equivalent content

Average whitespace words per parallel sentence:

- eng: 21.0
- hin: 24.7
- kan: 15.5
- tam: 16.2
- tel: 16.4

The Dravidian languages naturally pack more meaning into fewer whitespace-separated words, so tok/word ends up exaggerating fertility.

Also checked the report's claim that tok/word and tok/char "agree."

Correlation is about 0.75 on Hindi, which isn't that surprising since both metrics share the same numerator.

I think using tokens per parallel sentence relative to English is a much cleaner comparison.

---

## #8 Part B: capacity model

The formulae I used were from Google. Had to calculate by hand first to verify if they actually are real or if AI is hallucinating again.

Calculated KV capacity from the hardware assumptions:

24 × 0.92 GB
− 8.4 GB weights
− 1.6 GB overhead

≈12.08 GB available for KV cache.

Using 114,688 bytes/token (28 layers × 2 × 8 KV heads × 128 × 2 bytes), a 4096-token sequence needs about 0.47 GB.

Almost 25 sequences.

The nice surprise was that the predicted KV utilization matches the logged `kv_cache_util` almost perfectly before saturation (0.16, 0.31, 0.62, 0.93).

The predicted preemption point (between batch 24 and 32) is exactly where the logs show it.

---

## #9 Understanding `reported_tok_s`

Why is this so hard.
I suspected `reported_tok_s` included prompt (prefill) tokens.

Reconstructed it as:

`batch × (prompt + generation) / wall_time`

Matches every row within rounding.

That changes the interpretation quite a bit.

For example, at batch 16:

- reported throughput looks better for long prompts
- generation-only throughput is actually much lower

Likewise, the "3200 tok/s at batch 48" claim doesn't hold up once KV preemption is included.

---

## #10 Possible fixes

Two obvious options:

- FP8 KV cache halves KV memory (57,344 bytes/token), which should allow about 51 concurrent 4k sequences and comfortably fit batch 48.
- Simpler option: cap `max_num_seqs` at 24.

Added both to `partB/partB_solution.md` together with concrete predictions so they're easy to verify later.

---

## If I started earlier

- Compare against an Indic tokenizer (AI4Bharat or similar).
- Build a small code-mixed (Hinglish) evaluation set.
- Validate the FP8 prediction on an actual serving stack.
- Part C is still only a proposal. I couldn't run those experiments here, so every number is clearly marked as a prediction rather than a measurement.