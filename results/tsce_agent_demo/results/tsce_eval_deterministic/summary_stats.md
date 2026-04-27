| method | passes | CI95 |
|--------|--------|------|
| baseline   | 84/100 | 75.58% – 89.90% |
| tsce_t1    | 84/100 | 75.58% – 89.90% |
| tsce_diff  | 88/100 | 80.19% – 93.00% |

McNemar baseline vs tsce p = 0.683

McNemar baseline vs tsce_diff p = 0.221

McNemar tsce vs tsce_diff p = 0.221

Wilcoxon |baseline error| > |tsce error| p = 0.00253

Wilcoxon |baseline error| > |tsce_diff error| p = 0.00253
Silhouette (cosine) = nan
KL(unigram)  bits  = 3.255
Intrinsic-dim  B → T   14.27 → 13.99

Hull-area (t-SNE):
* baseline: 3130.363
* tsce_diff: 3119.857
* tsce_t1: 3092.563


### Models
* Baseline: gpt-4.1-mini
* TSCE anchor: gpt-4.1-mini-2025-04-14
* TSCE final: gpt-4.1-mini-2025-04-14
