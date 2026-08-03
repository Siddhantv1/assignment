

#analyze.py -- Part A3: corrected cross-language tokenization analysis
"""
Corpus: FLORES-200 dev, 997 parallel sentences per lang

English, Hindi, Tamil Telugu, Kannada

Fixes vs fertility.py :
  1. no lowercasing (it distorts cased languages, is a no-op for the rest)
  2. Changed to split(), so no empty strings from double spaces
  3. using aggregate ratios = (sum tokens / sum denominator) and not mean of per-line ratios, to save costs
  4. multiple denominators: whitespace word, grapheme cluster (regex \\X),
    UTF-8 byte, parallel sentence
  5. 2 tokenizers instead of only 1: gpt2 (English, byte-BPE) and xlm-roberta-base
    (multilingual, 250k vocab for other scripts)

"""

import sys
import os
import regex

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ASSIGNMENT"))
sys.path.insert(0, base_dir)

from starter_kit.starter_kit.fertility import load_tokenizer, read_lines

sys.stdout.reconfigure(encoding="utf-8")

FLORES = {
    "eng": "partA/corpus_flores/eng_Latn.dev",
    "hin": "partA/corpus_flores/hin_Deva.dev",
    "kan": "partA/corpus_flores/kan_Knda.dev",
    "tam": "partA/corpus_flores/tam_Taml.dev",
    "tel": "partA/corpus_flores/tel_Telu.dev",
}
TOKENIZERS = [("gpt2", "gpt2"), ("xlmr", "hf:xlm-roberta-base")]


def graphemes(s):
    return regex.findall(r"\X", s)


def main():
    corpora = {lang: read_lines(p) for lang, p in FLORES.items()}

    for tok_name, spec in TOKENIZERS:
        enc = load_tokenizer(spec)
        print(f"\n### tokenizer: {tok_name}")
        header = f"{'lang':<6}{'tok/word':>10}{'tok/grapheme':>14}{'tok/byte':>10}{'tok/sent':>10}"
        print(header)
        print("-" * len(header))
        totals = {}
        for lang, lines in corpora.items():
            t = sum(len(enc(l)) for l in lines)
            w = sum(len(l.split()) for l in lines)
            g = sum(len(graphemes(l)) for l in lines)
            b = sum(len(l.encode("utf-8")) for l in lines)
            s = len(lines)
            totals[lang] = (t, w, g, b, s)
            print(f"{lang:<6}{t / w:>10.3f}{t / g:>14.3f}{t / b:>10.3f}{t / s:>10.2f}")
        print("  ratios vs eng:")
        for lang in corpora:
            if lang == "eng":
                continue
            t, w, g, b, s = totals[lang]
            te, we, ge, be, se = totals["eng"]
            print(f"  {lang}/eng: tok/word {t / w / (te / we):5.2f}x   "
                  f"tok/grapheme {t / g / (te / ge):5.2f}x   "
                  f"tok/byte {t / b / (te / be):5.2f}x   "
                  f"tok/sent {t / s / (te / se):5.2f}x")

    print("\n### decision row: relative cost per unit of content")
    print("tok/sentence on parallel data = tokens to express the SAME content;")
    print("this is the number a routing/cost decision should use 👍")


if __name__ == "__main__":
    main()
