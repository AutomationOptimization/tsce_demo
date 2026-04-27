# Anchor Token Property Analysis

## Condition Summary
| condition | n | mean lift | proxy ratio | family coverage | avg token len |
|---|---:|---:|---:|---:|---:|
| anchor | 64 | 1.1629 | 0.2083 | 2.00 | 6.14 |
| anchor_head_deleted | 64 | 0.8332 | 0.2031 | 2.00 | 6.15 |
| anchor_mid_deleted | 64 | 1.0441 | 0.2131 | 2.00 | 6.14 |
| anchor_neutral_replaced | 64 | 0.8777 | 0.1651 | 2.00 | 6.11 |
| anchor_shuffled | 64 | 0.9398 | 0.1274 | 1.98 | 6.22 |
| anchor_tail_deleted | 64 | 1.0766 | 0.2093 | 1.98 | 6.14 |
| context_collision_anchor | 320 | 0.8269 | 0.0000 | 0.00 | 5.52 |
| random_matched_anchor_1 | 64 | -0.0593 | 0.0016 | 0.08 | 9.23 |
| random_matched_anchor_2 | 64 | -0.0011 | 0.0019 | 0.09 | 9.05 |
| random_matched_anchor_3 | 64 | -0.0231 | 0.0027 | 0.12 | 9.04 |

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
| only_sleep_quiet | 4 | 0.7203 | 0.2222 | 1.00 | 1.00 |
| random_neutral | 4 | -0.8351 | 0.0000 | 0.00 | 0.00 |

## Coverage Summary
| target family coverage | n | mean lift | proxy ratio |
|---:|---:|---:|---:|
| 0 | 4 | -0.8351 | 0.0000 |
| 1 | 16 | 0.8271 | 0.2222 |
| 2 | 24 | 1.4721 | 0.2222 |
| 3 | 16 | 1.3424 | 0.2222 |
| 4 | 4 | 1.9312 | 0.2222 |

## Family Presence Summary
| family | with n | with lift | without n | without lift | delta |
|---|---:|---:|---:|---:|---:|
| glass_light | 32 | 1.5253 | 32 | 0.8006 | 0.7247 |
| avian_sky | 32 | 1.5168 | 32 | 0.8090 | 0.7078 |
| garden_green | 32 | 1.3253 | 32 | 1.0006 | 0.3247 |
| sleep_quiet | 32 | 1.2335 | 32 | 1.0923 | 0.1412 |

## Feature Correlations With Lift
- `avg_token_len`: -0.2719
- `long_ratio`: -0.2708
- `target_family_coverage`: 0.2472
- `target_proxy_ratio`: 0.2388
- `token_count`: 0.2125
- `avian_sky_ratio`: 0.1860
- `short_ratio`: -0.1706
- `glass_light_ratio`: 0.1700
- `instruction_noise_ratio`: 0.1232
- `garden_green_ratio`: 0.1002
- `sleep_quiet_ratio`: 0.0763
- `unique_ratio`: -0.0583
- `story_magic_ratio`: -0.0191

## Anchor-Condition Correlations With Lift
- `unique_ratio`: 0.6663
- `target_proxy_ratio`: 0.6230
- `target_family_coverage`: 0.5731
- `avg_token_len`: -0.3392
- `avian_sky_ratio`: 0.2868
- `glass_light_ratio`: 0.2803
- `long_ratio`: -0.2222
- `sleep_quiet_ratio`: -0.0438
- `garden_green_ratio`: 0.0091
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
- `only_sleep_quiet_002` (only_sleep_quiet): anchor lift=0.8031, random-matched mean lift=0.1080, delta=0.6951
- `only_sleep_quiet_003` (only_sleep_quiet): anchor lift=0.6267, random-matched mean lift=-0.7284, delta=1.3551
- `only_sleep_quiet_004` (only_sleep_quiet): anchor lift=0.4599, random-matched mean lift=-0.5467, delta=1.0066
- `random_neutral_001` (random_neutral): anchor lift=-0.2174, random-matched mean lift=0.0957, delta=-0.3131
- `random_neutral_002` (random_neutral): anchor lift=-1.2949, random-matched mean lift=-0.3391, delta=-0.9558
- `random_neutral_003` (random_neutral): anchor lift=-0.9397, random-matched mean lift=0.3285, delta=-1.2682
- `random_neutral_004` (random_neutral): anchor lift=-0.8885, random-matched mean lift=0.0403, delta=-0.9288
