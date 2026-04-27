| method | passes | CI95 |
|--------|--------|------|
| baseline   | 89/100 | 81.37% – 93.75% |
| tsce_t1    | 88/100 | 80.19% – 93.00% |
| tsce_diff  | 87/100 | 79.02% – 92.24% |

McNemar baseline vs tsce p = 1

McNemar baseline vs tsce_diff p = 0.617

McNemar tsce vs tsce_diff p = 1

Wilcoxon |baseline error| > |tsce error| p = 0.00384

Wilcoxon |baseline error| > |tsce_diff error| p = 0.00384
Silhouette (cosine) = nan
KL(unigram)  bits  = 2.509
Intrinsic-dim  B → T   15.54 → 15.29

Hull-area (t-SNE):
* baseline: 34276.606
* tsce_diff: 35036.462
* tsce_t1: 34562.396


### Models
* Baseline: gpt-4.1-mini
* TSCE anchor: gpt-4.1-mini-2025-04-14
* TSCE final: gpt-4.1-mini-2025-04-14
