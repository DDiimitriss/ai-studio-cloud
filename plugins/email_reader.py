import os
import imaplib
import email
from email.header import decode_header

IMAP_SERVER = os.environ.get("IMAP_SERVER", "imap.gmail.com")
EMAIL_ACCOUNT = os.environ.get("SENDER_EMAIL", "")
EMAIL_PASSWORD = os.environ.get("SENDER_PASSWORD", "")

def run(args):
    folder = args.get("folder", "INBOX")
    limit = int(args.get("limit", 5))

    if not EMAIL_ACCOUNT or not EMAIL_PASSWORD:
        return "Error: Email credentials not set. Set SENDER_EMAIL and SENDER_PASSWORD."

    try:
        mail = imaplib.IMAP4_SSL(IMAP_SERVER)
        mail.login(EMAIL_ACCOUNT, EMAIL_PASSWORD)
        mail.select(folder)

        status, messages = mail.search(None, "ALL")
        email_ids = messages[0].split()[-limit:]

        result = []
        for e_id in reversed(email_ids):
            status, msg_data = mail.fetch(e_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject = decode_header(msg["Subject"])[0][0]
                    if isinstance(subject, bytes):
                        subject = subject.decode(errors="ignore")
                    from_ = decode_header(msg.get("From"))[0][0]
                    if isinstance(from_, bytes):
                        from_ = from_.decode(errors="ignore")
                    result.append(f"From: {from_}\nSubject: {subject}")
        mail.close()
        mail.logout()
        return "\n\n".join(result) if result else "No emails found."
    except Exception as e:
        return f"Error reading email: {e}"

def get_info():
    return {
        "name": "Email Reader",
        "description": "Reads recent emails from your inbox. Arguments: 'folder' (INBOX by default), 'limit' (number of emails, default 5). Use when user asks 'what emails did I get?' or 'show my latest emails'."
    }