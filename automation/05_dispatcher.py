"""
Email Dispatcher.

Sends nudge emails and the weekly ops report via SMTP.

Default target: Mailtrap sandbox (safe for portfolio demos — no real emails sent).
Swap SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASS in config.py or environment
variables to point at a real provider (SendGrid, Gmail, SES, etc.).

All sends are logged to automation/logs/dispatch_log.csv.

Portfolio note:
  Sign up at https://mailtrap.io → Inboxes → SMTP Settings.
  Copy the credentials into .streamlit/secrets.toml:
      SMTP_USER = "your-mailtrap-username"
      SMTP_PASS = "your-mailtrap-password"
  Visitors can then open the shared Mailtrap inbox URL to see every email
  that was "sent" — fully rendered HTML, no real delivery.
"""
import smtplib
import csv
import os
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from datetime             import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from automation.config import (
    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS,
    SENDER_NAME, SENDER_EMAIL, OPERATOR_EMAIL, LOGS_DIR,
)

LOG_PATH = os.path.join(LOGS_DIR, "dispatch_log.csv")
_LOG_HEADERS = ["timestamp", "recipient", "subject", "type", "status", "error"]


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

def _init_log():
    os.makedirs(LOGS_DIR, exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_LOG_HEADERS)


def _log_send(recipient: str, subject: str, msg_type: str,
              status: str, error: str = ""):
    _init_log()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.utcnow().isoformat(timespec="seconds"),
            recipient, subject, msg_type, status, error,
        ])


# ---------------------------------------------------------------------------
# Core send function
# ---------------------------------------------------------------------------

def _send(to: str, subject: str, html: str, msg_type: str) -> bool:
    """
    Send an HTML email.  Returns True on success, False on failure.
    If SMTP_USER is blank, skips sending and logs as 'skipped'
    (useful when credentials aren't configured yet).
    """
    if not SMTP_USER:
        print(f"[Dispatcher] SMTP not configured — skipping send to {to}")
        _log_send(to, subject, msg_type, "skipped")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"{SENDER_NAME} <{SENDER_EMAIL}>"
    msg["To"]      = to
    msg.attach(MIMEText(html, "html", "utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SENDER_EMAIL, to, msg.as_string())
        _log_send(to, subject, msg_type, "sent")
        print(f"[Dispatcher] Sent '{subject}' → {to}")
        return True
    except Exception as e:
        _log_send(to, subject, msg_type, "failed", str(e))
        print(f"[Dispatcher] ERROR sending to {to}: {e}")
        return False


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def send_driver_nudge(driver_email: str, nudge: dict) -> bool:
    """
    Send a single gamified CRM nudge to a driver.

    Args:
        driver_email: recipient address (use fake addresses for demo)
        nudge:        dict returned by nudge_engine.generate_nudge()
    """
    return _send(
        to       = driver_email,
        subject  = nudge["subject"],
        html     = nudge["html"],
        msg_type = f"nudge:{nudge['variant']}",
    )


def send_ops_report(report_html: str, recipient: str = OPERATOR_EMAIL) -> bool:
    """Send the weekly HTML ops report to the operator email address."""
    subject = f"Uber Pro — Weekly Ops Report ({datetime.utcnow().strftime('%Y-%m-%d')})"
    return _send(
        to       = recipient,
        subject  = subject,
        html     = report_html,
        msg_type = "ops_report",
    )


def load_dispatch_log() -> list[dict]:
    """Return all dispatch log entries as a list of dicts (newest first)."""
    _init_log()
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows   = list(reader)
    return list(reversed(rows))


if __name__ == "__main__":
    print("SMTP target:", SMTP_HOST, SMTP_PORT)
    print("SMTP user configured:", bool(SMTP_USER))
    print("Dispatch log:", LOG_PATH)
    log = load_dispatch_log()
    print(f"Log entries: {len(log)}")
