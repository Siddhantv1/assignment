# A4 — Recommendation memo: tokenizer cost & routing (1 page)


## Corrected headline numbers

Measured on 997 parallel FLORES-200 sentences per language, aggregate tokens
per parallel sentence relative to English (the content-controlled unit):

| | Hindi | Kannada | Tamil | Telugu |
|---|---|---|---|---|
| gpt2 tokenizer (as in v0) | 7.45× | 13.59× | 15.43× | 13.04× |
| multilingual tokenizer (xlm-roberta-base) | **1.26×** | **1.37×** | **1.35×** | **1.33×** |

REPORT_v0's "Hindi costs ~6× English" is dominated by a tokenizer artifact,
not a property of the Devanagari script: switching only the tokenizer moves
the Hindi number from 7.45× to 1.26×. (The v0 script also had smaller,
real bugs — asymmetric lowercasing, empty-string word counting, per-line
ratio averaging — each measured in `bug_exp.py`; together they move
the headline by a few percent, not the conclusion.)

## Recommendation

1. **Do not budget 6× serving cost for Hindi, and do not build a separate
   Indic tokenizer/model stack on the basis of v0.** The realistic planning
   number, if our served model uses a multilingual tokenizer, is **1.3–1.4×
   English cost per equivalent request** for Hindi and Dravidian languages.
2. **Make tokenizer vocabulary coverage a launch criterion, not a routing
   rule.** If any served model uses an English-centric tokenizer, Indic
   traffic genuinely costs 7–15× — the fix is the tokenizer, not the router.
3. Adopt **tokens per parallel sentence vs the English baseline** as the
   standard cross-language cost metric. "Fertility per word" should be
   retired for cross-language decisions: a whitespace word is not a constant
   unit of content across languages.

## Biggest caveat

The corpus is 997 professionally translated, formally written sentences. Our
real traffic will be shorter, messier, and heavily code-mixed (Hinglish etc.),
which both tokenizers handle worse and differently. These are *relative*
numbers on clean text; absolute production cost must be confirmed on real
traffic. Sample size is adequate for the ratio (CIs are tight), not for
tail behavior.

## Production metric to monitor

**Realized tokens per request (prompt + completion) by language, as a ratio
against the English baseline, on live traffic** — e.g. weekly p50 of
`total_tokens/request` per detected language. Alert if hin/eng drifts above
~1.5× or moves materially from the 1.26 baseline. This single number catches
every way this analysis can go wrong in production: a model/tokenizer swap
that regresses Indic coverage, a shift toward code-mixed traffic, or a
prompting change that inflates Indic responses.
