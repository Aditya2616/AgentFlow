## Fault dose-response: no verifier, retry only, full verifier

Success = task passes its checker. Intervals are 95% Wilson. Rate 0 is one clean run per arm (greedy decoding makes seeds identical); other rates use the seeds shown in `n`. Model calls and tool calls are means per task run.

### transient tool failures

| fault rate | arm | success [95% CI] | model calls | tool calls | faults injected |
|---|---|---|---|---|---|
| 0% | no verifier | 67% (16/24) [47, 82] | 3.9 | 1.8 | 0.0 |
| 0% | retry only | 67% (16/24) [47, 82] | 4.2 | 2.3 | 0.0 |
| 0% | full verifier | 67% (16/24) [47, 82] | 5.6 | 3.2 | 0.0 |
| 15% | no verifier | 53% (64/120) [44, 62] | 3.9 | 1.8 | 0.3 |
| 15% | retry only | 67% (80/120) [58, 74] | 4.1 | 2.6 | 0.4 |
| 15% | full verifier | 68% (81/120) [59, 75] | 5.5 | 3.6 | 0.5 |
| 30% | no verifier | 36% (43/120) [28, 45] | 3.9 | 1.8 | 0.7 |
| 30% | retry only | 65% (78/120) [56, 73] | 3.9 | 3.2 | 1.1 |
| 30% | full verifier | 68% (82/120) [60, 76] | 5.5 | 4.3 | 1.4 |
| 50% | no verifier | 36% (43/120) [28, 45] | 3.9 | 1.8 | 0.8 |
| 50% | retry only | 56% (67/120) [47, 64] | 3.7 | 3.0 | 1.3 |
| 50% | full verifier | 68% (82/120) [60, 76] | 6.0 | 5.0 | 2.4 |

Paired exact tests (same task and seed; discordant pairs only, two-sided sign test). Seeds of one task are not independent, so treat the p-values as optimistic.

| fault rate | comparison | only first passes | only second passes | exact p |
|---|---|---|---|---|
| 0% | retry only vs no verifier | 0 | 0 | 1.0000 |
| 0% | full verifier vs retry only | 1 | 1 | 1.0000 |
| 0% | full verifier vs no verifier | 1 | 1 | 1.0000 |
| 15% | retry only vs no verifier | 16 | 0 | 0.0000 |
| 15% | full verifier vs retry only | 5 | 4 | 1.0000 |
| 15% | full verifier vs no verifier | 21 | 4 | 0.0009 |
| 30% | retry only vs no verifier | 35 | 0 | 0.0000 |
| 30% | full verifier vs retry only | 7 | 3 | 0.3438 |
| 30% | full verifier vs no verifier | 42 | 3 | 0.0000 |
| 50% | retry only vs no verifier | 24 | 0 | 0.0000 |
| 50% | full verifier vs retry only | 17 | 2 | 0.0007 |
| 50% | full verifier vs no verifier | 41 | 2 | 0.0000 |

### silently wrong results

| fault rate | arm | success [95% CI] | model calls | tool calls | faults injected |
|---|---|---|---|---|---|
| 0% | no verifier | 67% (16/24) [47, 82] | 3.9 | 1.8 | 0.0 |
| 0% | retry only | 67% (16/24) [47, 82] | 4.2 | 2.3 | 0.0 |
| 0% | full verifier | 67% (16/24) [47, 82] | 5.6 | 3.2 | 0.0 |
| 15% | no verifier | 67% (80/120) [58, 74] | 3.9 | 1.8 | 0.0 |
| 15% | retry only | 67% (80/120) [58, 74] | 4.2 | 2.3 | 0.1 |
| 15% | full verifier | 67% (80/120) [58, 74] | 5.6 | 3.3 | 0.1 |
| 30% | no verifier | 67% (80/120) [58, 74] | 3.9 | 1.8 | 0.1 |
| 30% | retry only | 67% (80/120) [58, 74] | 4.2 | 2.3 | 0.1 |
| 30% | full verifier | 67% (80/120) [58, 74] | 5.7 | 3.4 | 0.1 |
| 50% | no verifier | 57% (69/120) [49, 66] | 3.9 | 1.8 | 0.2 |
| 50% | retry only | 57% (69/120) [49, 66] | 4.2 | 2.3 | 0.2 |
| 50% | full verifier | 62% (75/120) [54, 71] | 6.8 | 4.2 | 0.8 |

Paired exact tests (same task and seed; discordant pairs only, two-sided sign test). Seeds of one task are not independent, so treat the p-values as optimistic.

| fault rate | comparison | only first passes | only second passes | exact p |
|---|---|---|---|---|
| 0% | retry only vs no verifier | 0 | 0 | 1.0000 |
| 0% | full verifier vs retry only | 1 | 1 | 1.0000 |
| 0% | full verifier vs no verifier | 1 | 1 | 1.0000 |
| 15% | retry only vs no verifier | 0 | 0 | 1.0000 |
| 15% | full verifier vs retry only | 5 | 5 | 1.0000 |
| 15% | full verifier vs no verifier | 5 | 5 | 1.0000 |
| 30% | retry only vs no verifier | 0 | 0 | 1.0000 |
| 30% | full verifier vs retry only | 5 | 5 | 1.0000 |
| 30% | full verifier vs no verifier | 5 | 5 | 1.0000 |
| 50% | retry only vs no verifier | 0 | 0 | 1.0000 |
| 50% | full verifier vs retry only | 11 | 5 | 0.2101 |
| 50% | full verifier vs no verifier | 11 | 5 | 0.2101 |

