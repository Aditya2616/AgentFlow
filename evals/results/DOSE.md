## Fault dose-response: no verifier, retry only, full verifier

Success = task passes its checker. Intervals are 95% Wilson. Rate 0 is one clean run per arm (greedy decoding makes seeds identical); other rates use the seeds shown in `n`. Model calls and tool calls are means per task run.

### transient tool failures

| fault rate | arm | success [95% CI] | model calls | tool calls | faults injected |
|---|---|---|---|---|---|
| 0% | no verifier | 67% (16/24) [47, 82] | 3.9 | 1.8 | 0.0 |
| 0% | retry only | 67% (16/24) [47, 82] | 4.2 | 2.3 | 0.0 |
| 0% | full verifier | 62% (15/24) [43, 79] | 8.3 | 5.8 | 0.0 |
| 15% | no verifier | 53% (64/120) [44, 62] | 3.9 | 1.8 | 0.3 |
| 15% | retry only | 67% (80/120) [58, 74] | 4.1 | 2.6 | 0.4 |

Paired exact tests (same task and seed; discordant pairs only, two-sided sign test). Seeds of one task are not independent, so treat the p-values as optimistic.

| fault rate | comparison | only first passes | only second passes | exact p |
|---|---|---|---|---|
| 0% | retry only vs no verifier | 0 | 0 | 1.0000 |
| 0% | full verifier vs retry only | 0 | 1 | 1.0000 |
| 0% | full verifier vs no verifier | 0 | 1 | 1.0000 |
| 15% | retry only vs no verifier | 16 | 0 | 0.0000 |

### silently wrong results

| fault rate | arm | success [95% CI] | model calls | tool calls | faults injected |
|---|---|---|---|---|---|
| 0% | no verifier | 67% (16/24) [47, 82] | 3.9 | 1.8 | 0.0 |
| 0% | retry only | 67% (16/24) [47, 82] | 4.2 | 2.3 | 0.0 |
| 0% | full verifier | 62% (15/24) [43, 79] | 8.3 | 5.8 | 0.0 |

Paired exact tests (same task and seed; discordant pairs only, two-sided sign test). Seeds of one task are not independent, so treat the p-values as optimistic.

| fault rate | comparison | only first passes | only second passes | exact p |
|---|---|---|---|---|
| 0% | retry only vs no verifier | 0 | 0 | 1.0000 |
| 0% | full verifier vs retry only | 0 | 1 | 1.0000 |
| 0% | full verifier vs no verifier | 0 | 1 | 1.0000 |

