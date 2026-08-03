# Part A: Tokenizer Audit results

## A1: The eval Corpus building
I have used the FLORES 200 dev split corpus, which has the 977 parallel sentences per language. 

Source:
 [The official NLLB dist](https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz)

 Paper: https://arxiv.org/abs/2207.04672

Downloaded using the `fetch_corpus.py` script, and no preprocessing was done except NFC.

Languages: English, Hindi, Kannada, Tamil & Telugu
(eng_Latn, hin_Deva, kan_Knda, tam_Taml, tel_Telu)

**What the corpus cannot tell us:**
1. The 977 sentences from Wikipedia's news/health/travel text is okay for relative tokenizer comparison, not for our traffic. Real user chats are short and sometimes ambiguous, with mix in 2 languages (Hinglish or Tanglish) and every tokenizer handles it differently. This corpus hasn't sampled it.
   
2. It also cannot tell us about the downstream quality, the tokenizer being cheap for every sentence does not mean it can't harm the model.

3. The numbers here should be read as the same content relative cost instead of absolute production cost.




## A2: Auditing the `fertility.py` script.


### For this part I ran experiments using a script `bug_exp.py`

| Experiment | Claim | Verdict | Evidence (before vs after) | Direction, impact |
|---|---|---|---|---|
| E1 | `line.split(" ")` counts empty strings as "words" when a line has double spaces | **Bug** | Sample corpora each have 1/10 lines with double spaces; fertility 1.269 → 1.253 (eng), 7.525 → 7.403 (hin) | Deflates fertility ~1.3–1.6% on the toy corpus; unbounded on messy whitespace |
| E2 | Unconditional `.lower()` before tokenizing | **Bug for cross-language comparison** | FLORES eng tokens: 25,741 → 26,696 lowercased (**+3.7%**); hin: +0.01% (Devanagari has no case). hin/eng per-sentence ratio: 7.45 without vs 7.19 with `.lower()` | Asymmetric by construction; silently moves the headline ratio by ~3.6% |
| E3 | Averages per-line ratios instead of aggregate (sum tokens / sum words) | **Bug for a cost metric** | FLORES: mean-of-ratios vs aggregate = 1.237 vs 1.228 (eng), 7.825 vs 7.796 (hin) | Small here (+0.4–0.7%) but wrong in principle: a cost model needs totals; per-line averaging lets short lines outvote long ones |
| E4 | `unicodedata.normalize("NFC", …)` | **Suspicious-looking, actually fine** | 90/997 FLORES hin lines aren't NFC; corpus-wide tokens 191,589 → 191,828 with NFC (+0.12%, negligible); worst-case fully-NFD nukta string: 49 tokens with and without gpt2 | Negligible; and it's defensive — canonical input matches what tokenizers were trained on. **Not a bug.** |
| E5 | "5.89× is a property of the script, not the tokenizer" (report's root cause) | **Wrong — it's mostly the tokenizer** | tok/parallel-sentence on gpt2 vs xlm-roberta-base: hin 192.4 → 36.7; kan 350.8 → 39.7; tam 398.4 → 39.2. hin/eng ratio: **7.45× on gpt2 vs 1.26× on a multilingual tokenizer** | The single largest distortion in the whole report: ~6× |
| E6 | *Conceptual flaw*: "per whitespace word" holds nothing constant across languages | **The metric says the wrong thing** | Whitespace-words per *same-content* sentence: eng 21.0, hin 24.7, kan 15.5, tam 16.2, tel 16.4 | Agglutinative Dravidian languages pack a sentence into ~16 "words"; dividing tokens by that mechanically inflates their fertility even when token cost is identical |
| E7 | Report: "tok/char agrees with fertility, so the result is robust" | **Not independent confirmation** | Per-line Pearson r(tok/word, tok/char) on hin = 0.75 | Both metrics share the same numerator (token count); the denominators (words, chars) are correlated. Agreement is mechanical, not corroboration |

**Note**
Also, it was found that the line `random.seed(1337)` isn't useful, it does nothing. So it was just a harmless import of `random` module, not a bug.



## A3: Corrected Analysis, using `analyze.py`

The corrected analysis applies the following fixes:
1. Removing the lowercasing of tokens
2. using the correct `split()` for words splitting
3. Using Aggregate Ratios
4. **Using 4 denominators**: whitespace word, grapheme cluster, parallel sentence, UTF-8 byte
5. **Using 2 tokenizers**: gpt2 (English centric and uses byte-BPE), xlm-roberta-base = multilingual SentencePiece with 250k vocab that covers all 5 scripts


The results of analysis is stored in `analyze.txt` as well

### Results:
```
### tokenizer: gpt2
lang    tok/word  tok/grapheme  tok/byte  tok/sent
--------------------------------------------------
eng        1.228         0.206     0.205     25.82
hin        7.796         2.328     0.595    192.41
kan       22.668         4.059     0.979    350.82
tam       24.617         4.204     0.996    398.36
tel       20.481         4.562     0.991    336.65
  ratios vs eng:
  hin/eng: tok/word  6.35x   tok/grapheme 11.32x   tok/byte  2.89x   tok/sent  7.45x
  kan/eng: tok/word 18.45x   tok/grapheme 19.74x   tok/byte  4.76x   tok/sent 13.59x
  tam/eng: tok/word 20.04x   tok/grapheme 20.45x   tok/byte  4.85x   tok/sent 15.43x
  tel/eng: tok/word 16.67x   tok/grapheme 22.19x   tok/byte  4.82x   tok/sent 13.04x

### tokenizer: xlmr
lang    tok/word  tok/grapheme  tok/byte  tok/sent
--------------------------------------------------
eng        1.384         0.232     0.231     29.08
hin        1.489         0.445     0.114     36.74
kan        2.567         0.460     0.111     39.72
tam        2.423         0.414     0.098     39.21
tel        2.362         0.526     0.114     38.82
  ratios vs eng:
  hin/eng: tok/word  1.08x   tok/grapheme  1.92x   tok/byte  0.49x   tok/sent  1.26x
  kan/eng: tok/word  1.85x   tok/grapheme  1.98x   tok/byte  0.48x   tok/sent  1.37x
  tam/eng: tok/word  1.75x   tok/grapheme  1.79x   tok/byte  0.42x   tok/sent  1.35x
  tel/eng: tok/word  1.71x   tok/grapheme  2.27x   tok/byte  0.49x   tok/sent  1.33x

### decision row: relative cost per unit of content
tok/sentence on parallel data = tokens to express the SAME content;
this is the number a routing/cost decision should use 👍
```

   
### Q: What single number should drive a routing-and-cost decision, and why? 

Tokens per parallel sentence relative to English is the denominator that should drive that decision. 
It is because it holds the **content** across all languages constantly. 

Why not others:

- The whitespace "word" is different unit of meaning in Tamil as compared to English (E6)
  
- Graphene cluster is unit of script, for single characters made with 1 or more letters. (base letter + diacritic)
  
- UTF-8 byte is only a unit of encoding. The Indic scripts are 3 Byte/char, therefore tok/byte  makes it look nice under xlmr (0.49x)

The per request cost increases with total tokens for same sentence, therefore tok/sentence works here, on parallel data.
The other denominators are good for tokenizer mechanics but they only help in compressing and not the request costs.


On a multilingual tokenizer:
- Hindi costs 1.26x more than English per unit of content
  
- Dravidian Languages cost ~ 1.3 to 1.4x (NOT 6x)
