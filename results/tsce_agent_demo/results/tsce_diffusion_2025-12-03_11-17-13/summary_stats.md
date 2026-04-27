| method | passes | CI95 |
|--------|--------|------|
| baseline   | 291/300 | 94.40% – 98.41% |
| tsce_diff  | 293/300 | 95.26% – 98.87% |

McNemar baseline vs tsce_diff p = 0.724

Wilcoxon |baseline error| > |tsce_diff error| p = 0.673
Silhouette (cosine) = 0.005
KL(unigram)  bits  = 1.634
Intrinsic-dim  B → tsce_diff   21.02 → 21.65

Hull-area (t-SNE):
* baseline: 4139.765
* tsce_diff: 4242.609
* tsce_diff_anchor: 354.650


### Models
* Baseline: gpt-4.1-mini
* TSCE anchor: N/A
* TSCE final: N/A
