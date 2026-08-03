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




