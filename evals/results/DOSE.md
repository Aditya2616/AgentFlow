## Fault dose-response: no verifier, retry only, full verifier

Success = task passes its checker. Intervals are 95% Wilson. Rate 0 is one clean run per arm (greedy decoding makes seeds identical); other rates use the seeds shown in `n`. Model calls and tool calls are means per task run.

### transient tool failures

| fault rate | arm | success [95% CI] | model calls | tool calls | faults injected |
|---|---|---|---|---|---|
| 0% | no verifier | 67% (16/24) [47, 82] | 3.9 | 1.8 | 0.0 |
| 0% | retry only | 67% (16/24) [47, 82] | 4.2 | 2.3 | 0.0 |
| 0% | full verifier | 67% (16/24) [47, 82] | 5.6 | 3.2 | 0.0 |
| 15% | no verifier | 60% (72/120) [51, 68] | 3.9 | 1.8 | 0.2 |
| 15% | retry only | 67% (80/120) [58, 74] | 4.1 | 2.5 | 0.2 |
| 15% | full verifier | 68% (81/120) [59, 75] | 5.5 | 3.4 | 0.3 |
| 30% | no verifier | 48% (58/120) [40, 57] | 3.9 | 1.8 | 0.4 |
| 30% | retry only | 63% (76/120) [54, 71] | 4.0 | 2.8 | 0.7 |
| 30% | full verifier | 67% (80/120) [58, 74] | 5.5 | 3.9 | 0.9 |
| 50% | no verifier | 35% (42/120) [27, 44] | 3.9 | 1.8 | 0.8 |
| 50% | retry only | 50% (60/120) [41, 59] | 3.5 | 3.0 | 1.4 |
| 50% | full verifier | 62% (74/120) [53, 70] | 5.9 | 5.0 | 2.3 |

Paired exact tests (same task and seed; discordant pairs only, two-sided sign test). Seeds of one task are not independent, so treat the p-values as optimistic.

| fault rate | comparison | only first passes | only second passes | exact p |
|---|---|---|---|---|
| 0% | retry only vs no verifier | 0 | 0 | 1.0000 |
| 0% | full verifier vs retry only | 1 | 1 | 1.0000 |
| 0% | full verifier vs no verifier | 1 | 1 | 1.0000 |
| 15% | retry only vs no verifier | 8 | 0 | 0.0078 |
| 15% | full verifier vs retry only | 5 | 4 | 1.0000 |
| 15% | full verifier vs no verifier | 13 | 4 | 0.0490 |
| 30% | retry only vs no verifier | 18 | 0 | 0.0000 |
| 30% | full verifier vs retry only | 8 | 4 | 0.3877 |
| 30% | full verifier vs no verifier | 26 | 4 | 0.0001 |
| 50% | retry only vs no verifier | 18 | 0 | 0.0000 |
| 50% | full verifier vs retry only | 16 | 2 | 0.0013 |
| 50% | full verifier vs no verifier | 34 | 2 | 0.0000 |

### silently wrong results

| fault rate | arm | success [95% CI] | model calls | tool calls | faults injected |
|---|---|---|---|---|---|
| 0% | no verifier | 67% (16/24) [47, 82] | 3.9 | 1.8 | 0.0 |
| 0% | retry only | 67% (16/24) [47, 82] | 4.2 | 2.3 | 0.0 |
| 0% | full verifier | 67% (16/24) [47, 82] | 5.6 | 3.2 | 0.0 |
| 15% | no verifier | 60% (72/120) [51, 68] | 3.9 | 1.8 | 0.2 |
| 15% | retry only | 62% (75/120) [54, 71] | 4.1 | 2.3 | 0.2 |
| 15% | full verifier | 67% (80/120) [58, 74] | 5.8 | 3.5 | 0.2 |
| 30% | no verifier | 52% (63/120) [44, 61] | 3.9 | 1.8 | 0.3 |
| 30% | retry only | 55% (66/120) [46, 64] | 4.1 | 2.3 | 0.3 |
| 30% | full verifier | 66% (79/120) [57, 74] | 6.2 | 3.8 | 0.6 |
| 50% | no verifier | 42% (50/120) [33, 51] | 3.9 | 1.8 | 0.6 |
| 50% | retry only | 45% (54/120) [36, 54] | 4.1 | 2.3 | 0.5 |
| 50% | full verifier | 61% (73/120) [52, 69] | 7.1 | 4.6 | 1.2 |

Paired exact tests (same task and seed; discordant pairs only, two-sided sign test). Seeds of one task are not independent, so treat the p-values as optimistic.

| fault rate | comparison | only first passes | only second passes | exact p |
|---|---|---|---|---|
| 0% | retry only vs no verifier | 0 | 0 | 1.0000 |
| 0% | full verifier vs retry only | 1 | 1 | 1.0000 |
| 0% | full verifier vs no verifier | 1 | 1 | 1.0000 |
| 15% | retry only vs no verifier | 3 | 0 | 0.2500 |
| 15% | full verifier vs retry only | 9 | 4 | 0.2668 |
| 15% | full verifier vs no verifier | 12 | 4 | 0.0768 |
| 30% | retry only vs no verifier | 3 | 0 | 0.2500 |
| 30% | full verifier vs retry only | 17 | 4 | 0.0072 |
| 30% | full verifier vs no verifier | 20 | 4 | 0.0015 |
| 50% | retry only vs no verifier | 4 | 0 | 0.1250 |
| 50% | full verifier vs retry only | 22 | 3 | 0.0002 |
| 50% | full verifier vs no verifier | 26 | 3 | 0.0000 |

