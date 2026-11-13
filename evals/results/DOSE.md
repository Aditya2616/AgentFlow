## Fault dose-response: no verifier, retry only, full verifier

Success = task passes its checker. Intervals are 95% Wilson. Rate 0 is one clean run per arm (greedy decoding makes seeds identical); other rates use the seeds shown in `n`. Model calls and tool calls are means per task run.

### transient tool failures

| fault rate | arm | success [95% CI] | model calls | tool calls | faults injected |
|---|---|---|---|---|---|
| 0% | no verifier | 63% (15/24) [43, 79] | 3.9 | 1.8 | 0.0 |
| 0% | retry only | 63% (15/24) [43, 79] | 4.2 | 2.3 | 0.0 |
| 0% | full verifier | 71% (17/24) [51, 85] | 5.6 | 3.2 | 0.0 |
| 15% | no verifier | 58% (70/120) [49, 67] | 3.9 | 1.8 | 0.2 |
| 15% | retry only | 63% (75/120) [54, 71] | 4.1 | 2.5 | 0.2 |
| 15% | full verifier | 71% (85/120) [62, 78] | 5.5 | 3.4 | 0.3 |
| 30% | no verifier | 48% (57/120) [39, 56] | 3.9 | 1.8 | 0.4 |
| 30% | retry only | 59% (71/120) [50, 68] | 4.0 | 2.8 | 0.7 |
| 30% | full verifier | 70% (84/120) [61, 77] | 5.5 | 3.9 | 0.9 |
| 50% | no verifier | 35% (42/120) [27, 44] | 3.9 | 1.8 | 0.8 |
| 50% | retry only | 48% (57/120) [39, 56] | 3.5 | 3.0 | 1.4 |
| 50% | full verifier | 63% (76/120) [54, 71] | 5.9 | 5.0 | 2.3 |

Paired exact tests (same task and seed; discordant pairs only, two-sided sign test). Seeds of one task are not independent, so treat the p-values as optimistic.

| fault rate | comparison | only first passes | only second passes | exact p |
|---|---|---|---|---|
| 0% | retry only vs no verifier | 0 | 0 | 1.0000 |
| 0% | full verifier vs retry only | 2 | 0 | 0.5000 |
| 0% | full verifier vs no verifier | 2 | 0 | 0.5000 |
| 15% | retry only vs no verifier | 7 | 2 | 0.1797 |
| 15% | full verifier vs retry only | 10 | 0 | 0.0020 |
| 15% | full verifier vs no verifier | 15 | 0 | 0.0001 |
| 30% | retry only vs no verifier | 16 | 2 | 0.0013 |
| 30% | full verifier vs retry only | 13 | 0 | 0.0002 |
| 30% | full verifier vs no verifier | 27 | 0 | 0.0000 |
| 50% | retry only vs no verifier | 17 | 2 | 0.0007 |
| 50% | full verifier vs retry only | 19 | 0 | 0.0000 |
| 50% | full verifier vs no verifier | 34 | 0 | 0.0000 |

### silently wrong results

| fault rate | arm | success [95% CI] | model calls | tool calls | faults injected |
|---|---|---|---|---|---|
| 0% | no verifier | 63% (15/24) [43, 79] | 3.9 | 1.8 | 0.0 |
| 0% | retry only | 63% (15/24) [43, 79] | 4.2 | 2.3 | 0.0 |
| 0% | full verifier | 71% (17/24) [51, 85] | 5.6 | 3.2 | 0.0 |
| 15% | no verifier | 57% (68/120) [48, 65] | 3.9 | 1.8 | 0.2 |
| 15% | retry only | 59% (71/120) [50, 68] | 4.1 | 2.3 | 0.2 |
| 15% | full verifier | 71% (85/120) [62, 78] | 5.8 | 3.5 | 0.2 |
| 30% | no verifier | 50% (60/120) [41, 59] | 3.9 | 1.8 | 0.3 |
| 30% | retry only | 53% (63/120) [44, 61] | 4.1 | 2.3 | 0.3 |
| 30% | full verifier | 71% (85/120) [62, 78] | 6.3 | 3.9 | 0.6 |
| 50% | no verifier | 40% (48/120) [32, 49] | 3.9 | 1.8 | 0.6 |
| 50% | retry only | 43% (52/120) [35, 52] | 4.1 | 2.3 | 0.5 |
| 50% | full verifier | 66% (79/120) [57, 74] | 7.2 | 4.7 | 1.2 |

Paired exact tests (same task and seed; discordant pairs only, two-sided sign test). Seeds of one task are not independent, so treat the p-values as optimistic.

| fault rate | comparison | only first passes | only second passes | exact p |
|---|---|---|---|---|
| 0% | retry only vs no verifier | 0 | 0 | 1.0000 |
| 0% | full verifier vs retry only | 2 | 0 | 0.5000 |
| 0% | full verifier vs no verifier | 2 | 0 | 0.5000 |
| 15% | retry only vs no verifier | 3 | 0 | 0.2500 |
| 15% | full verifier vs retry only | 14 | 0 | 0.0001 |
| 15% | full verifier vs no verifier | 17 | 0 | 0.0000 |
| 30% | retry only vs no verifier | 3 | 0 | 0.2500 |
| 30% | full verifier vs retry only | 22 | 0 | 0.0000 |
| 30% | full verifier vs no verifier | 25 | 0 | 0.0000 |
| 50% | retry only vs no verifier | 4 | 0 | 0.1250 |
| 50% | full verifier vs retry only | 27 | 0 | 0.0000 |
| 50% | full verifier vs no verifier | 31 | 0 | 0.0000 |

