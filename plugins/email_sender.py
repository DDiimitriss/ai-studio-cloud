import os
import smtplib
from email.message import EmailMessage

SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587))
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "")
SENDER_PASSWORD = os.environ.get("SENDER_PASSWORD", "")

def run(args):
    to = args.get("to")
    subject = args.get("subject", "Message from AI")
    body = args.get("body", "")

    if not to:
        return "Error: missing 'to' recipient"
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        return "Error: email credentials not set. Set SENDER_EMAIL and SENDER_PASSWORD."

    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = SENDER_EMAIL
        msg["To"] = to

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        return f"✅ Email sent to {to}"
    except Exception as e:
        return f"❌ Error sending email: {e}"

def get_info():
    return {
        "name": "Email Sender",
        "description": "Sends an email. Use when user asks to send, email, or message someone. Arguments: 'to' (required), 'subject', 'body'."
    }