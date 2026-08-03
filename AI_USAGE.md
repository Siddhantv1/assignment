# AI_USAGE.md

For this assignment, I used the Kimi K3 model with Kimi Code CLI harness for coding, Gemini for research and topic-understanding. Following is the honest breakdown.

## Work done by AI

- **Code:** AI wrote ~80% of `fetch_corpus.py`, `bug_exp.py`, `analyze.py`, and parts of `capacity.py`. I specified what each experiment had to isolate; AI chose the implementation.

- **Analysis drafts:** early drafts of documentations `partA_solution.md`, `partB_solution.md`, both memos, were AI-written. I edited most of it then reviewed every numeric claim against the actual script output files (`experiment_result.txt`,
  `analyze.txt`, `capacity_results.txt`) — every number in the writeups is traceable to a command.

- **Debugging:** AI diagnosed the corpus-download dead ends (moved GitHub data, gated HF mirrors), path errors, Windows console-encoding crashes.

## Where AI was wrong or misleading (and got corrected by measurement)

- AI's initial hypothesis was that `.lower()` would *decrease* GPT-2's English token count. Measured: it *increases* it by 3.71%. The written
  claim uses the measured direction, not the hypothesis.

- AI expected the NFD worst-case string to tokenize into more tokens under gpt2. Measured: identical (49 = 49). The E4 verdict ("suspicious but fine") rests on the measurement, not the intuition.

- AI hallucinated that FLORES-200 lived in the facebookresearch/flores GitHub repo.
  The working URL was recovered from a mirror dataset's loading script.

## What I verified independently

- Re-ran the original `fertility.py` and matched REPORT_v0's numbers.
- Researched the numeric-wise formulae, then solved it by hand before writing to `capacity.py`
- Re-ran all three analysis scripts and confirmed the outputs quoted in the writeups match the committed output files.
- Hand-checked the KV-cache arithmetic (B1) and the  `n×(prompt+gen)/wall = reported_tok_s` reconstruction (B3) with a
  calculator for two rows each.

## Things I am still learning to understand deeply

- The internals of SentencePiece vs byte-level BPE training (I can explain what they do to fertility, not the training algorithms).
- vLLM's preemption implementation details beyond what's needed for B2/B4.
- Part C is reasoning under constraints, not executed work; the arithmetic is mine to defend but the GPU-hours estimates are unvalidated.

## Rule I followed
