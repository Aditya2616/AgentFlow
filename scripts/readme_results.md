Everything below is Qwen3-0.6B on a GPU in float16 (the GPU is only a speed choice; the code runs the same on CPU), the
24-task suite, greedy decoding. Model replies are cached by prompt, so the three arms share replies wherever their
prompts match, which makes **model calls** the fair cost measure here, not wall-clock seconds. Reproduce with
`agentflow ablate --rates 0,0.15,0.3,0.5 --seeds 5` followed by `agentflow eval-report`. Result files are under
`evals/results/`: `dose/*.json` (one per cell), `DOSE.md` (full tables with paired tests), `FAILURES.md`, `SUMMARY.md`.

**Three arms.** *No verifier*: every result is passed on unchecked. *Retry only*: a tool call that raises is retried
with the error text (same limit of 3 tries), but nothing else is checked and there is no re-plan; when the retries run
out the run fails. *Full verifier*: result checks, retries and re-plans, plus the email send guard.

**Two kinds of injected fault.** *Transient*: the tool call raises a timeout. *Silent*: the tool succeeds but returns a
well-formed wrong result (no search hits, zero SQL rows, empty Python output, or a document cut to 40 characters).
Rates are per tool call; each (task, seed) pair has its own seeded fault stream. Rate 0 is one run per arm (greedy
decoding makes the seeds identical); other rates are 24 tasks x 5 seeds = 120 runs per arm.

![Task success against fault rate](docs/assets/dose.svg)

The same chart, drawn from `results.json`, is the Results tab of the UI.

{{dose}}

What the numbers say, with the caveats that matter:

* **Clean tasks: the verifier does not help.** All three arms pass 16 of 24 (67%), and seven tasks fail in every arm.
* **Transient failures are mostly a retry story.** Retry-only holds up to 30% (63% against 67% for the full verifier,
  paired exact p = 0.39, not significant) while no verifier falls to 48%. Only at 50% does the full verifier pull ahead
  of retry-only (62% against 50%, p = 0.0013), because it can re-plan once the retries are used up.
* **Silent wrong results are where checking matters.** Retry-only is statistically indistinguishable from no verifier
  (its retries never fire, nothing raised). The full verifier is ahead of retry-only by 11 points at 30% (66% against
  55%, p = 0.007) and 16 points at 50% (61% against 45%, p = 0.0002); at 15% the gap is not significant (p = 0.27).
* **It costs model calls.** At 50% silent faults the full verifier uses 7.1 model calls per task against 4.1 for
  retry-only.
* **Not everything is caught.** A cut-off document is a well-formed non-empty result and the verifier has no check for
  it. The silent faults it recovers are the empty ones.
* The p-values are paired sign tests over (task, seed) pairs. Seeds of one task are not independent, so they are
  optimistic. The 24 tasks are my own and small.

### Where the clean run fails

Per-task view of the clean run (no faults), with a category assigned by rules from the saved traces
(`src/agentflow/evals/breakdown.py`), not by hand:

{{clean}}

Seven tasks fail in every arm: four because the 0.6B planner leaves out a step the task needs (for example it plans
only a Python step for a question that needs a database lookup first), and three because it cannot write working
arguments (an unknown column, an undefined variable, or no tool call at all) within the retry budget. More tries do not fix
a model that does not know the schema. Two tasks differ between arms: t06 passes only with the full verifier (its
re-plan fixes a bad first plan), and t23 fails only with it, and that one is the checker, not the agent: the agent
safely reported it could not find the file but worded it as "not provided in the results", which is not in the
checker's list of refusal phrases. I left the checker as it was.

### The pointless email (t23) and the send guard

In the first run the model sent an email to an invented address on a task that asked for none. The send guard is a
check on side-effecting steps: an email step is dropped from the plan when the request names no recipient, and an
email whose address the request does not name is rejected before anything is written. Measured on the clean run:

{{extra}}

Without the guard the model wrote 12 emails, 5 of them to invented addresses on tasks (t01, t06, t09, t13, t15)
that asked for none; with it, 7 emails were written and all 7 are on tasks that name an address. The one task that
changed from pass to fail (t23) is the checker's phrase list, described above, not an email. The guard is a rule about
the request text, not a judgement: it would block a legitimate send to an address the user did not type.

### Larger model

Qwen2.5-1.5B-Instruct (about 3 GB in float16) was run on the clean suite only, before the send guard existed, so it is
not comparable with the Qwen3-0.6B rows in every detail. It passed 13 of 24 without the verifier and 14 of 24 with it:
bigger did not mean better here. Its first, uncached pass took about 30 seconds per task on a shared GPU.
