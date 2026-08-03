

# bug_exp.py, for Part A2: audit of fertility.py
"""
running 1 experiment per flaw i claim.


Claims under test:
  E1  line.split(" ") counts empty strings as words on double spaces (BUG)

  E2  unconditional .lower() changes token counts, asymmetrically across
      languages (BUG for a cross-language comparison)

  E3  mean of per-line ratios != aggregate ratio (weights every line
      equally regardless of length) (BUG for a cost metric)

  E4  unicodedata.normalize("NFC", ...) -- looks suspicious, is actually
      fine (defensive, near no-op on clean corpora)

  E5  the 5.89x Hindi/English gap is mostly a GPT-2 artifact, not "a
      property of the script" (report's root-cause claim is wrong)

  E6  conceptual: "per whitespace word" does not hold content constant
      across languages; agglutinative languages are penalized mechanically

  E7  report claim "tok/char confirms fertility" -- the two metrics share
      a numerator, so agreement is mechanical, not independent confirmation

"""

import statistics
import sys
import os
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from starter_kit.starter_kit.fertility import load_tokenizer, read_lines


def resolve_path(*parts):
    return str(REPO_ROOT.joinpath(*parts))


SAMPLE = {
    "eng": resolve_path("starter_kit", "starter_kit", "corpus_sample", "eng_sample.txt"),
    "hin": resolve_path("starter_kit", "starter_kit", "corpus_sample", "hin_sample.txt"),
}
FLORES = {
    "eng": resolve_path("partA", "corpus_flores", "eng_Latn.dev"),
    "hin": resolve_path("partA", "corpus_flores", "hin_Deva.dev"),
    "kan": resolve_path("partA", "corpus_flores", "kan_Knda.dev"),
    "tam": resolve_path("partA", "corpus_flores", "tam_Taml.dev"),
    "tel": resolve_path("partA", "corpus_flores", "tel_Telu.dev"),
}


def rule(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def e1_double_space(encode):
    rule("E1: split(' ') counts empty strings as words (double spaces)")
    demo = "Please keep the books  in the cupboard."
    print(f"demo line: {demo!r}")
    print(f"  .split(' ') -> {demo.split(' ')}  ({len(demo.split(' '))} 'words')")
    print(f"  .split()    -> {demo.split()}  ({len(demo.split())} words)")
    for lang, path in SAMPLE.items():
        lines = read_lines(path)
        hit = sum(1 for l in lines if "  " in l)
        toks = sum(len(encode(l.lower())) for l in lines)
        w_buggy = sum(len(l.split(" ")) for l in lines)
        w_fixed = sum(len(l.split()) for l in lines)
        print(f"  {lang}: {hit}/10 lines contain double spaces; "
              f"fertility split(' ')={toks / w_buggy:.3f} vs "
              f"split()={toks / w_fixed:.3f} "
              f"(bug deflates fertility by {100 * (1 - (toks / w_buggy) / (toks / w_fixed)):.1f}%)")


def e2_lowercase(encode):
    rule("E2: unconditional .lower() distorts English token counts")
    for lang in ["eng", "hin"]:
        lines = read_lines(FLORES[lang])
        t_orig = sum(len(encode(l)) for l in lines)
        t_low = sum(len(encode(l.lower())) for l in lines)
        print(f"  {lang} (FLORES dev, gpt2): tokens original={t_orig}, "
              f"lowercased={t_low}, delta={100 * (t_low - t_orig) / t_orig:+.2f}%")
    lines_e = read_lines(FLORES["eng"])
    lines_h = read_lines(FLORES["hin"])
    def ratio(low):
        te = sum(len(encode(l.lower() if low else l)) for l in lines_e)
        th = sum(len(encode(l.lower() if low else l)) for l in lines_h)
        return (th / len(lines_h)) / (te / len(lines_e))
    print(f"  hin/eng tokens-per-sentence ratio: as-coded(.lower())={ratio(True):.3f} "
          f"vs no-lower={ratio(False):.3f}")


def e3_mean_of_ratios(encode):
    rule("E3: mean of per-line ratios vs aggregate (sum/sum)")
    for lang in ["eng", "hin"]:
        lines = read_lines(FLORES[lang])
        per_line = [len(encode(l)) / len(l.split()) for l in lines]
        mean_ratio = sum(per_line) / len(per_line)
        agg = sum(len(encode(l)) for l in lines) / sum(len(l.split()) for l in lines)
        print(f"  {lang}: mean-of-ratios={mean_ratio:.3f} vs aggregate={agg:.3f} "
              f"({100 * (mean_ratio - agg) / agg:+.1f}% distortion)")


def e4_nfc(encode):
    rule("E4: NFC normalization -- suspicious-looking but fine")
    lines = read_lines(FLORES["hin"])
    raw = [l for l in open(FLORES["hin"], encoding="utf-8") if l.strip()]
    raw = [l.strip() for l in raw]
    diff = [l for l in raw if unicodedata.normalize("NFC", l) != l]
    t_nfc = sum(len(encode(unicodedata.normalize("NFC", l))) for l in raw)
    t_raw = sum(len(encode(l)) for l in raw)
    print(f"  {len(diff)}/{len(raw)} FLORES hin lines are not NFC; corpus-wide "
          f"tokens with/without normalize: {t_nfc} vs {t_raw} "
          f"({100 * (t_nfc - t_raw) / t_raw:+.2f}% -- negligible)")
    if diff:
        l = diff[0]
        t_before, t_after = len(encode(l)), len(encode(unicodedata.normalize("NFC", l)))
        print(f"  example differing line: {l[:40]}...  tokens {t_before} -> {t_after}")
    # worst case: fully decomposed (NFD) input, e.g. nukta words like फ़िल्म
    nfd = unicodedata.normalize("NFD", "फ़िल्म क़ानून मुश्किल ज़रूरत")
    nfc = unicodedata.normalize("NFC", nfd)
    print(f"  worst-case NFD input {nfd!r}: NFD tokens={len(encode(nfd))} vs "
          f"NFC tokens={len(encode(nfc))} (gpt2) -- normalizing bounds the damage")


def e5_tokenizer_choice(encode_gpt2, encode_xlmr):
    rule("E5: is the 5.89x gap 'a property of the script'? Compare tokenizers")
    for lang in ["eng", "hin", "kan", "tam", "tel"]:
        lines = read_lines(FLORES[lang])
        for name, enc in [("gpt2", encode_gpt2), ("xlmr", encode_xlmr)]:
            tps = sum(len(enc(l)) for l in lines) / len(lines)
            print(f"  {lang} {name:>5}: {tps:7.2f} tok/sentence", end="")
        print()
    for name, enc in [("gpt2", encode_gpt2), ("xlmr", encode_xlmr)]:
        r = {}
        for lang in ["eng", "hin"]:
            lines = read_lines(FLORES[lang])
            r[lang] = sum(len(enc(l)) for l in lines) / len(lines)
        print(f"  hin/eng tok-per-sentence ratio with {name:>5}: {r['hin'] / r['eng']:.2f}x")


def e6_words_not_constant():
    rule("E6: 'per whitespace word' does not hold content constant")
    print("  (parallel sentences = same content; count words per sentence)")
    for lang, path in FLORES.items():
        lines = read_lines(path)
        wps = sum(len(l.split()) for l in lines) / len(lines)
        print(f"  {lang}: {wps:.2f} whitespace-words per parallel sentence")


def e7_shared_numerator(encode):
    rule("E7: tok/word and tok/char share a numerator -- agreement is mechanical")
    lines = read_lines(FLORES["hin"])
    xs, ys = [], []
    for l in lines:
        t = len(encode(l))
        xs.append(t / len(l.split()))
        ys.append(t / len(l))
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    r = cov / (statistics.pstdev(xs) * statistics.pstdev(ys) * len(xs))
    print(f"  per-line Pearson r(tok/word, tok/char) on hin FLORES = {r:.3f}")
    print("  -> the two 'independent' metrics are the same tokens divided by")
    print("     two correlated denominators; agreement confirms nothing")


def main():
    encode_gpt2 = load_tokenizer("gpt2")
    print("loading xlm-roberta-base (multilingual baseline)...")
    encode_xlmr = load_tokenizer("hf:xlm-roberta-base")
    e1_double_space(encode_gpt2)
    e2_lowercase(encode_gpt2)
    e3_mean_of_ratios(encode_gpt2)
    e4_nfc(encode_gpt2)
    e5_tokenizer_choice(encode_gpt2, encode_xlmr)
    e6_words_not_constant()
    e7_shared_numerator(encode_gpt2)


if __name__ == "__main__":
    main()
