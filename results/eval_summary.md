30 episodes per setting, deterministic policy, max episode length 300

| agent | stage | setting | return_mean | return_std | length_mean | success_% | crash_% | idle_% | speed_mps | distance_m |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 0 | empty track | -2.5 | 1.8 | 10.7 | 0.0 | 100.0 | 0.0 | 9.4 | 19.9 |
| random | 1 | one car | -2.4 | 2.0 | 10.6 | 0.0 | 100.0 | 0.0 | 9.3 | 20.0 |
| random | 2 | light traffic | -2.9 | 1.7 | 12.9 | 0.0 | 100.0 | 0.0 | 8.8 | 21.0 |
| random | 3 | heavy traffic | -3.0 | 1.5 | 10.4 | 0.0 | 100.0 | 0.0 | 9.2 | 19.0 |
| A_curriculum_a0.2 | 0 | empty track | 375.7 | 22.7 | 297.0 | 96.7 | 3.3 | 0.0 | 19.6 | 1162.9 |
| A_curriculum_a0.2 | 1 | one car | 331.6 | 110.6 | 266.9 | 83.3 | 16.7 | 0.0 | 19.4 | 1042.6 |
| A_curriculum_a0.2 | 2 | light traffic | 223.5 | 146.8 | 188.1 | 40.0 | 60.0 | 0.0 | 18.9 | 728.8 |
| A_curriculum_a0.2 | 3 | heavy traffic | 29.5 | 29.3 | 38.0 | 0.0 | 100.0 | 0.0 | 16.7 | 133.5 |
| B_curriculum_a0.05 | 0 | empty track | 321.8 | 104.7 | 265.6 | 83.3 | 16.7 | 0.0 | 19.5 | 1041.7 |
| B_curriculum_a0.05 | 1 | one car | 330.8 | 83.9 | 275.5 | 86.7 | 13.3 | 0.0 | 19.5 | 1079.4 |
| B_curriculum_a0.05 | 2 | light traffic | 201.6 | 135.6 | 178.8 | 36.7 | 63.3 | 0.0 | 18.9 | 692.0 |
| B_curriculum_a0.05 | 3 | heavy traffic | 29.9 | 21.8 | 40.5 | 0.0 | 100.0 | 0.0 | 16.8 | 139.7 |
| C_no_curriculum_a0.2 | 0 | empty track | 372.5 | 3.2 | 300.0 | 100.0 | 0.0 | 0.0 | 19.7 | 1180.7 |
| C_no_curriculum_a0.2 | 1 | one car | 335.4 | 87.5 | 274.8 | 86.7 | 13.3 | 0.0 | 19.5 | 1077.6 |
| C_no_curriculum_a0.2 | 2 | light traffic | 284.4 | 122.0 | 240.4 | 63.3 | 36.7 | 0.0 | 19.1 | 936.8 |
| C_no_curriculum_a0.2 | 3 | heavy traffic | 28.6 | 24.3 | 37.5 | 0.0 | 100.0 | 0.0 | 16.9 | 130.3 |
| D_curriculum_a0.2_seed1 | 0 | empty track | 389.4 | 2.5 | 300.0 | 100.0 | 0.0 | 0.0 | 19.6 | 1178.6 |
| D_curriculum_a0.2_seed1 | 1 | one car | 362.7 | 76.3 | 285.2 | 93.3 | 6.7 | 0.0 | 19.5 | 1117.6 |
| D_curriculum_a0.2_seed1 | 2 | light traffic | 264.2 | 141.9 | 218.5 | 53.3 | 46.7 | 0.0 | 19.0 | 849.3 |
| D_curriculum_a0.2_seed1 | 3 | heavy traffic | 27.3 | 22.9 | 36.5 | 0.0 | 100.0 | 0.0 | 16.7 | 126.8 |
