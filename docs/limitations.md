# Limitations (read before citing a score)

- **Deterministic metrics are proxies.** token F1, ROUGE-L and BLEU measure surface overlap, not meaning. `bleu` here is BLEU-2 with a brevity penalty, not the 4-gram BLEU of sacreBLEU, so scores are not comparable with published BLEU.
- **`faithfulness` is lexical.** It measures how much of the answer's wording is supported by the context, not whether the claim is true; a negation or a swapped entity can still score high.
- **`number_match`** compares canonical decimal values (thousands separators removed, trailing zeros normalised). It does not understand units, scales ("2.2 million") or dates.
- **`no_refusal` and `safety_flag` are pattern lists.** They miss paraphrased refusals and unsafe content they do not list.
- **LLM-as-judge** results depend on the judge model and prompt and are not calibrated against human labels in this repository.
- **Small benchmark.** The bundled 18-case RAG benchmark is hand-written by the author and is for demonstration, not evidence about any model.
- **Verified by:** unit tests (including known-value checks for token F1, ROUGE-L and BLEU-2, and regression tests for two defects found in a verification pass: identical stop-word-only text scored 0 for token F1, and `number_match` split numbers with thousands separators).
