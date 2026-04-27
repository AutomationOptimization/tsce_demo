# Gemma 4 Evaluation Report

## Summary
- `supports_human_opaque_control`: not_yet
- `supports_soft_prompt_wording`: not_yet
- TSCE pass rate: 0.6600
- Random valid pass rate: 0.6500
- Shuffled anchor pass rate: 0.6400
- Context-collision pass rate: 0.6600
- English control pass rate: 0.6400

## Condition Table
| Condition | Count | Pass Rate | Mean Score | Mean Latency (s) | Mean Logit Shift | Mean Attention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 100 | 0.6600 | 0.6600 | 12.2893 | n/a | n/a |
| english_control | 100 | 0.6400 | 0.6400 | 10.5149 | n/a | n/a |
| tsce_anchor | 100 | 0.6600 | 0.6600 | 11.6629 | n/a | n/a |
| random_valid_anchor | 100 | 0.6500 | 0.6500 | 11.7403 | n/a | n/a |
| shuffled_anchor | 100 | 0.6400 | 0.6400 | 12.0669 | n/a | n/a |
| context_collision_anchor | 100 | 0.6600 | 0.6600 | 12.0899 | n/a | n/a |

## Interpretation
- `supports_human_opaque_control` turns `yes` only when TSCE beats shuffled and valid-anchor opaque controls while also showing non-zero internal movement.
- `supports_soft_prompt_wording` remains conservative and only upgrades when TSCE also beats the English control path.