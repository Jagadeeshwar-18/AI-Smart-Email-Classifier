from src.gmail.gmail_auth import get_gmail_service
import base64

def fetch_emails(max_emails=1000):
    service = get_gmail_service()
    emails = []

    request = service.users().messages().list(userId="me", labelIds=["INBOX"], maxResults=500)

    while request and len(emails) < max_emails:
        response = request.execute()

        for msg in response.get("messages", []):
            data = service.users().messages().get(
                userId="me", id=msg["id"], format="full"
            ).execute()

            headers = data["payload"].get("headers", [])
            subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")

            body = ""
            parts = data["payload"].get("parts", [])
            for p in parts:
                if p["mimeType"] == "text/plain":
                    body = base64.urlsafe_b64decode(p["body"]["data"]).decode("utf-8", errors="ignore")
                    break

            emails.append({"subject": subject, "body": body})

        request = service.users().messages().list_next(request, response)

    return emails
