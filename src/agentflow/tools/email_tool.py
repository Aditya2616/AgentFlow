"""Email tool. Writes to a local outbox; sends over SMTP only with explicit configuration.

Real sending requires BOTH AGENTFLOW_SMTP_URL (smtp://user:pass@host:port) and
AGENTFLOW_ALLOW_SEND=1. With either missing, the message only lands in the outbox folder.
"""

from __future__ import annotations

import json
import re
import smtplib
import time
import uuid
from email.message import EmailMessage
from pathlib import Path
from urllib.parse import unquote, urlparse

ADDR = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(\.[A-Za-z0-9\-]+)+$")


class EmailError(Exception):
    pass


def valid_address(addr: str) -> bool:
    return bool(ADDR.match(addr.strip()))


def run_email(
    outbox: Path,
    to: str,
    subject: str,
    body: str,
    *,
    smtp_url: str = "",
    allow_send: bool = False,
    allowed_domains: list[str] | None = None,
) -> dict:
    to = to.strip()
    if not valid_address(to):
        raise EmailError(f"{to!r} is not a valid email address")
    if allowed_domains and to.split("@")[1].lower() not in {d.lower() for d in allowed_domains}:
        raise EmailError(f"recipient domain is not on the allow-list ({', '.join(allowed_domains)})")
    if not subject.strip() or not body.strip():
        raise EmailError("subject and body must both be non-empty")

    msg = EmailMessage()
    msg["To"], msg["From"], msg["Subject"] = to, "agentflow@localhost", subject.strip()
    msg.set_content(body)

    outbox.mkdir(parents=True, exist_ok=True)
    mid = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    (outbox / f"{mid}.eml").write_bytes(bytes(msg))
    sent = False
    if smtp_url and allow_send:
        u = urlparse(smtp_url)
        with smtplib.SMTP(u.hostname, u.port or 587, timeout=10) as s:
            s.starttls()
            if u.username:
                s.login(unquote(u.username), unquote(u.password or ""))
            s.send_message(msg)
        sent = True
    entry = {"id": mid, "to": to, "subject": subject.strip(), "body": body, "sent": sent}
    with (outbox / "index.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    return {"outbox_id": mid, "to": to, "subject": subject.strip(), "sent_over_smtp": sent}
