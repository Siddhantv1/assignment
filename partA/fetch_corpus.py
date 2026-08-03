# script to fetch the FLORES 200 Corpus
"""
Languages: English, Hindi, Kannada, Tamil, Telugu


Preprocessing phase:
when analize script runs, empty lines get skipped. 

all the files are listed in parallel, helping us use the
"tokens per parallel sentence" as the denominator in analysis.


"""


import io
import os
import tarfile
import urllib.request

URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
LANGS = ["eng_Latn", "hin_Deva", "kan_Knda", "tam_Taml", "tel_Telu"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "corpus_flores")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    wanted = {f"./flores200_dataset/dev/{lang}.dev": lang for lang in LANGS}
    print(f"downloading {URL} ...")
    with urllib.request.urlopen(URL) as resp:
        blob = resp.read()
    print(f"got {len(blob) / 1e6:.1f} MB; extracting dev splits")
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        for member in tf.getmembers():
            if member.name in wanted:
                lang = wanted[member.name]
                data = tf.extractfile(member).read().decode("utf-8")
                out = os.path.join(OUT_DIR, f"{lang}.dev")
                with open(out, "w", encoding="utf-8") as f:
                    f.write(data)
                n = sum(1 for line in data.splitlines() if line.strip())
                print(f"  {lang}: {n} sentences -> {out}")


if __name__ == "__main__":
    main()