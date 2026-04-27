| method | passes | CI95 |
|--------|--------|------|
| baseline   | 27/30 | 74.38% – 96.54% |
| tsce_diff  | 28/30 | 78.68% – 98.15% |

McNemar baseline vs tsce_diff p = 1

Wilcoxon |baseline error| > |tsce_diff error| p = 0.5
Silhouette (cosine) = -0.015
KL(unigram)  bits  = 3.274
Intrinsic-dim  B → tsce_diff   12.59 → 12.61

Hull-area (t-SNE):
* baseline: 88.052
* tsce_diff: 100.760
* tsce_diff_anchor: 20.736


### Models
* Baseline: gpt-4.1-mini
* TSCE anchor: N/A
* TSCE final: N/A
