# Anchor Token Property Analysis

## Condition Summary
| condition | n | mean lift | proxy ratio | family coverage | avg token len |
|---|---:|---:|---:|---:|---:|
| anchor | 57 | 1.3312 | 0.2222 | 2.19 | 6.14 |
| anchor_head_deleted | 57 | 0.9757 | 0.2172 | 2.19 | 6.14 |
| anchor_mid_deleted | 57 | 1.2042 | 0.2272 | 2.19 | 6.13 |
| anchor_neutral_replaced | 57 | 0.9985 | 0.1769 | 2.19 | 6.11 |
| anchor_shuffled | 57 | 1.1214 | 0.1371 | 2.18 | 6.21 |
| anchor_tail_deleted | 57 | 1.2358 | 0.2220 | 2.18 | 6.13 |
| context_collision_anchor | 285 | 0.8297 | 0.0000 | 0.00 | 5.52 |
| random_matched_anchor_1 | 57 | -0.0534 | 0.0014 | 0.07 | 9.23 |
| random_matched_anchor_2 | 57 | 0.0115 | 0.0021 | 0.11 | 9.02 |
| random_matched_anchor_3 | 57 | 0.0030 | 0.0027 | 0.12 | 9.01 |

## Controlled Family Groups
| family group | n | mean lift | proxy ratio | observed coverage | controlled coverage |
|---|---:|---:|---:|---:|---:|
| all_four | 4 | 1.9312 | 0.2222 | 4.00 | 4.00 |
| coverage_2_avian_sky_garden_green | 4 | 1.3835 | 0.2222 | 2.00 | 2.00 |
| coverage_2_avian_sky_sleep_quiet | 4 | 0.9232 | 0.2222 | 2.00 | 2.00 |
| coverage_2_garden_green_sleep_quiet | 4 | 1.1761 | 0.2222 | 2.00 | 2.00 |
| coverage_2_glass_light_avian_sky | 4 | 2.2404 | 0.2222 | 2.00 | 2.00 |
| coverage_2_glass_light_garden_green | 4 | 1.6501 | 0.2222 | 2.00 | 2.00 |
| coverage_2_glass_light_sleep_quiet | 4 | 1.4595 | 0.2222 | 2.00 | 2.00 |
| coverage_3_avian_sky_garden_green_sleep_quiet | 4 | 1.3447 | 0.2222 | 3.00 | 3.00 |
| coverage_3_glass_light_avian_sky_garden_green | 4 | 1.7116 | 0.2222 | 3.00 | 3.00 |
| coverage_3_glass_light_avian_sky_sleep_quiet | 4 | 1.4028 | 0.2222 | 3.00 | 3.00 |
| coverage_3_glass_light_garden_green_sleep_quiet | 4 | 0.9103 | 0.2222 | 3.00 | 3.00 |
| only_avian_sky | 4 | 1.1974 | 0.2222 | 1.00 | 1.00 |
| only_garden_green | 4 | 0.4948 | 0.2222 | 1.00 | 1.00 |
| only_glass_light | 4 | 0.8961 | 0.2222 | 1.00 | 1.00 |
| only_sleep_quiet | 1 | 0.9913 | 0.2222 | 1.00 | 1.00 |

## Feature Correlations With Lift
- `long_ratio`: -0.2924
- `avg_token_len`: -0.2872
- `target_family_coverage`: 0.2387
- `target_proxy_ratio`: 0.2344
- `token_count`: 0.2183
- `short_ratio`: -0.1812
- `avian_sky_ratio`: 0.1789
- `glass_light_ratio`: 0.1611
- `sleep_quiet_ratio`: 0.0983
- `instruction_noise_ratio`: 0.0945
- `garden_green_ratio`: 0.0857
- `unique_ratio`: -0.0387
- `story_magic_ratio`: -0.0243

## Anchor-Condition Correlations With Lift
- `target_family_coverage`: 0.3356
- `avg_token_len`: -0.2465
- `garden_green_ratio`: -0.2197
- `unique_ratio`: 0.2175
- `avian_sky_ratio`: 0.1637
- `glass_light_ratio`: 0.1497
- `sleep_quiet_ratio`: -0.1193
- `long_ratio`: -0.1110
- `target_proxy_ratio`: -0.0000
- `story_magic_ratio`: 0.0000
- `instruction_noise_ratio`: 0.0000
- `short_ratio`: 0.0000
- `token_count`: 0.0000

## Per Anchor
- `all_four_001` (all_four): anchor lift=2.0063, random-matched mean lift=0.1207, delta=1.8856
- `all_four_002` (all_four): anchor lift=2.0113, random-matched mean lift=-0.0583, delta=2.0696
- `all_four_003` (all_four): anchor lift=1.8790, random-matched mean lift=0.1792, delta=1.6998
- `all_four_004` (all_four): anchor lift=1.8283, random-matched mean lift=-0.3943, delta=2.2226
- `coverage_2_avian_sky_garden_green_001` (coverage_2_avian_sky_garden_green): anchor lift=1.3752, random-matched mean lift=-0.2835, delta=1.6587
- `coverage_2_avian_sky_garden_green_002` (coverage_2_avian_sky_garden_green): anchor lift=1.5927, random-matched mean lift=-0.5463, delta=2.1390
- `coverage_2_avian_sky_garden_green_003` (coverage_2_avian_sky_garden_green): anchor lift=0.9049, random-matched mean lift=0.6356, delta=0.2693
- `coverage_2_avian_sky_garden_green_004` (coverage_2_avian_sky_garden_green): anchor lift=1.6611, random-matched mean lift=-0.0034, delta=1.6645
- `coverage_2_avian_sky_sleep_quiet_001` (coverage_2_avian_sky_sleep_quiet): anchor lift=0.8669, random-matched mean lift=0.0684, delta=0.7985
- `coverage_2_avian_sky_sleep_quiet_002` (coverage_2_avian_sky_sleep_quiet): anchor lift=1.1617, random-matched mean lift=-0.5703, delta=1.7320
- `coverage_2_avian_sky_sleep_quiet_003` (coverage_2_avian_sky_sleep_quiet): anchor lift=0.4071, random-matched mean lift=0.3284, delta=0.0787
- `coverage_2_avian_sky_sleep_quiet_004` (coverage_2_avian_sky_sleep_quiet): anchor lift=1.2570, random-matched mean lift=-0.4515, delta=1.7085
- `coverage_2_garden_green_sleep_quiet_001` (coverage_2_garden_green_sleep_quiet): anchor lift=1.3281, random-matched mean lift=0.1525, delta=1.1756
- `coverage_2_garden_green_sleep_quiet_002` (coverage_2_garden_green_sleep_quiet): anchor lift=1.1900, random-matched mean lift=0.0012, delta=1.1888
- `coverage_2_garden_green_sleep_quiet_003` (coverage_2_garden_green_sleep_quiet): anchor lift=0.8067, random-matched mean lift=0.0386, delta=0.7682
- `coverage_2_garden_green_sleep_quiet_004` (coverage_2_garden_green_sleep_quiet): anchor lift=1.3796, random-matched mean lift=0.4874, delta=0.8922
- `coverage_2_glass_light_avian_sky_001` (coverage_2_glass_light_avian_sky): anchor lift=3.5256, random-matched mean lift=-0.5281, delta=4.0537
- `coverage_2_glass_light_avian_sky_002` (coverage_2_glass_light_avian_sky): anchor lift=1.7202, random-matched mean lift=0.5541, delta=1.1661
- `coverage_2_glass_light_avian_sky_003` (coverage_2_glass_light_avian_sky): anchor lift=1.0207, random-matched mean lift=0.3714, delta=0.6493
- `coverage_2_glass_light_avian_sky_004` (coverage_2_glass_light_avian_sky): anchor lift=2.6951, random-matched mean lift=0.1236, delta=2.5715
- `coverage_2_glass_light_garden_green_001` (coverage_2_glass_light_garden_green): anchor lift=2.6804, random-matched mean lift=-0.4531, delta=3.1336
- `coverage_2_glass_light_garden_green_002` (coverage_2_glass_light_garden_green): anchor lift=1.4104, random-matched mean lift=-0.2555, delta=1.6659
- `coverage_2_glass_light_garden_green_003` (coverage_2_glass_light_garden_green): anchor lift=1.8122, random-matched mean lift=-0.1756, delta=1.9878
- `coverage_2_glass_light_garden_green_004` (coverage_2_glass_light_garden_green): anchor lift=0.6974, random-matched mean lift=0.3768, delta=0.3206
- `coverage_2_glass_light_sleep_quiet_001` (coverage_2_glass_light_sleep_quiet): anchor lift=2.8744, random-matched mean lift=-0.0858, delta=2.9602
- `coverage_2_glass_light_sleep_quiet_002` (coverage_2_glass_light_sleep_quiet): anchor lift=1.4195, random-matched mean lift=0.4570, delta=0.9625
- `coverage_2_glass_light_sleep_quiet_003` (coverage_2_glass_light_sleep_quiet): anchor lift=0.8403, random-matched mean lift=-0.2305, delta=1.0708
- `coverage_2_glass_light_sleep_quiet_004` (coverage_2_glass_light_sleep_quiet): anchor lift=0.7040, random-matched mean lift=0.0286, delta=0.6754
- `coverage_3_avian_sky_garden_green_sleep_quiet_001` (coverage_3_avian_sky_garden_green_sleep_quiet): anchor lift=1.4640, random-matched mean lift=-0.5162, delta=1.9802
- `coverage_3_avian_sky_garden_green_sleep_quiet_002` (coverage_3_avian_sky_garden_green_sleep_quiet): anchor lift=0.9885, random-matched mean lift=0.8664, delta=0.1221
- `coverage_3_avian_sky_garden_green_sleep_quiet_003` (coverage_3_avian_sky_garden_green_sleep_quiet): anchor lift=1.7507, random-matched mean lift=-0.3337, delta=2.0844
- `coverage_3_avian_sky_garden_green_sleep_quiet_004` (coverage_3_avian_sky_garden_green_sleep_quiet): anchor lift=1.1755, random-matched mean lift=0.2420, delta=0.9335
- `coverage_3_glass_light_avian_sky_garden_green_001` (coverage_3_glass_light_avian_sky_garden_green): anchor lift=2.4160, random-matched mean lift=0.3381, delta=2.0779
- `coverage_3_glass_light_avian_sky_garden_green_002` (coverage_3_glass_light_avian_sky_garden_green): anchor lift=1.9015, random-matched mean lift=0.2586, delta=1.6429
- `coverage_3_glass_light_avian_sky_garden_green_003` (coverage_3_glass_light_avian_sky_garden_green): anchor lift=0.8716, random-matched mean lift=-0.7544, delta=1.6260
- `coverage_3_glass_light_avian_sky_garden_green_004` (coverage_3_glass_light_avian_sky_garden_green): anchor lift=1.6573, random-matched mean lift=0.0629, delta=1.5944
- `coverage_3_glass_light_avian_sky_sleep_quiet_001` (coverage_3_glass_light_avian_sky_sleep_quiet): anchor lift=0.7586, random-matched mean lift=0.0109, delta=0.7477
- `coverage_3_glass_light_avian_sky_sleep_quiet_002` (coverage_3_glass_light_avian_sky_sleep_quiet): anchor lift=1.5212, random-matched mean lift=-0.8917, delta=2.4128
- `coverage_3_glass_light_avian_sky_sleep_quiet_003` (coverage_3_glass_light_avian_sky_sleep_quiet): anchor lift=1.7631, random-matched mean lift=-0.4108, delta=2.1739
- `coverage_3_glass_light_avian_sky_sleep_quiet_004` (coverage_3_glass_light_avian_sky_sleep_quiet): anchor lift=1.5685, random-matched mean lift=-0.3416, delta=1.9101
- `coverage_3_glass_light_garden_green_sleep_quiet_001` (coverage_3_glass_light_garden_green_sleep_quiet): anchor lift=0.5722, random-matched mean lift=0.1807, delta=0.3914
- `coverage_3_glass_light_garden_green_sleep_quiet_002` (coverage_3_glass_light_garden_green_sleep_quiet): anchor lift=0.3353, random-matched mean lift=0.0439, delta=0.2913
- `coverage_3_glass_light_garden_green_sleep_quiet_003` (coverage_3_glass_light_garden_green_sleep_quiet): anchor lift=1.2131, random-matched mean lift=0.2395, delta=0.9736
- `coverage_3_glass_light_garden_green_sleep_quiet_004` (coverage_3_glass_light_garden_green_sleep_quiet): anchor lift=1.5206, random-matched mean lift=0.1453, delta=1.3753
- `only_avian_sky_001` (only_avian_sky): anchor lift=0.9822, random-matched mean lift=-0.1620, delta=1.1442
- `only_avian_sky_002` (only_avian_sky): anchor lift=1.4844, random-matched mean lift=-0.0030, delta=1.4874
- `only_avian_sky_003` (only_avian_sky): anchor lift=1.2426, random-matched mean lift=-0.4098, delta=1.6525
- `only_avian_sky_004` (only_avian_sky): anchor lift=1.0803, random-matched mean lift=-0.1590, delta=1.2393
- `only_garden_green_001` (only_garden_green): anchor lift=0.3810, random-matched mean lift=-0.0734, delta=0.4544
- `only_garden_green_002` (only_garden_green): anchor lift=0.5562, random-matched mean lift=-0.4575, delta=1.0138
- `only_garden_green_003` (only_garden_green): anchor lift=0.4357, random-matched mean lift=0.6141, delta=-0.1784
- `only_garden_green_004` (only_garden_green): anchor lift=0.6061, random-matched mean lift=0.2348, delta=0.3713
- `only_glass_light_001` (only_glass_light): anchor lift=0.1819, random-matched mean lift=0.8216, delta=-0.6398
- `only_glass_light_002` (only_glass_light): anchor lift=0.9384, random-matched mean lift=0.1122, delta=0.8262
- `only_glass_light_003` (only_glass_light): anchor lift=1.4739, random-matched mean lift=-0.0327, delta=1.5066
- `only_glass_light_004` (only_glass_light): anchor lift=0.9901, random-matched mean lift=-0.4997, delta=1.4898
- `only_sleep_quiet_001` (only_sleep_quiet): anchor lift=0.9913, random-matched mean lift=0.2474, delta=0.7439
