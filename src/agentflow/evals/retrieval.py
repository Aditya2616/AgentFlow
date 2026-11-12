"""Retrieval check for the search tool: recall@1 and recall@3 on hand-labelled questions.

    python -m agentflow.evals.retrieval            # BM25 and hybrid (needs sentence-transformers)
"""

from __future__ import annotations

import json
import sys

from ..config import CORPUS_DIR
from ..tools.search import CorpusIndex

# (question, document that contains the answer)
QUERIES = [
    ("How long do I have to ask for a refund?", "refund_policy.md"),
    ("Can digital purchases be refunded?", "refund_policy.md"),
    ("How are refunds paid out?", "refund_policy.md"),
    ("How many vacation days do employees get?", "leave_policy.md"),
    ("Can unused holiday days roll over?", "leave_policy.md"),
    ("When is a doctor's note needed for sick leave?", "leave_policy.md"),
    ("What happens on a new hire's first day?", "onboarding.md"),
    ("By when must security training be finished?", "onboarding.md"),
    ("What is expected by day 60?", "onboarding.md"),
    ("How long must a password be?", "security_policy.md"),
    ("Is multi-factor authentication required?", "security_policy.md"),
    ("How fast must a security incident be reported?", "security_policy.md"),
    ("How did revenue change in the third quarter?", "q3_report.md"),
    ("Which region was strongest last quarter?", "q3_report.md"),
    ("What is the warranty on the Delta?", "products.md"),
    ("Which tracker has GPS?", "products.md"),
    ("Who owns the launch checklist?", "meeting_2026-10-14.md"),
    ("When is the November launch?", "meeting_2026-10-14.md"),
    ("When should an outage be escalated?", "outage_runbook.md"),
    ("What do I do first when the site is down?", "outage_runbook.md"),
    ("Is shipping free?", "shipping_policy.md"),
    ("How long does international delivery take?", "shipping_policy.md"),
    ("What are the payment terms?", "billing_faq.md"),
    ("What is the late payment fee?", "billing_faq.md"),
    ("How quickly do priority 1 tickets get a reply?", "support_sla.md"),
    ("What is the daily meal allowance when travelling?", "expense_policy.md"),
    ("When do I need approval for an expense?", "expense_policy.md"),
    ("What is the target for churn in Q4?", "okr_q4.md"),
    ("How long is the battery supplier lead time?", "vendor_notes.md"),
    ("What ships in the second half of 2027?", "roadmap_2027.md"),
]


def evaluate(mode: str) -> dict:
    idx = CorpusIndex(CORPUS_DIR, mode)
    at1 = at3 = 0
    for q, doc in QUERIES:
        hits = idx.search(q, k=3)
        docs = [h["doc"] for h in hits]
        at1 += bool(docs) and docs[0] == doc
        at3 += doc in docs
    n = len(QUERIES)
    return {"mode": mode, "n": n, "recall@1": round(at1 / n, 3), "recall@3": round(at3 / n, 3)}


if __name__ == "__main__":
    results = [evaluate("bm25")]
    try:
        results.append(evaluate("hybrid"))
    except Exception as exc:  # sentence-transformers or the model may be missing
        print(f"hybrid skipped: {exc}", file=sys.stderr)
    print(json.dumps(results, indent=1))
