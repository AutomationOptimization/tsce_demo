| method | passes | CI95 |
|--------|--------|------|
| baseline   | 292/300 | 94.83% – 98.64% |
| tsce_diff  | 290/300 | 93.97% – 98.18% |

McNemar baseline vs tsce_diff p = 0.724

Wilcoxon |baseline error| > |tsce_diff error| p = 0.5
Silhouette (cosine) = 0.006
KL(unigram)  bits  = 1.555
Intrinsic-dim  B → tsce_diff   20.42 → 23.51

Hull-area (t-SNE):
* baseline: 3662.173
* tsce_diff: 3876.770
* tsce_diff_anchor: 353.054


### Models
* Baseline: gpt-4.1-mini
* TSCE anchor: N/A
* TSCE final: N/A
