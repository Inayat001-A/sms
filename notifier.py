import os
import smtplib
from email.message import EmailMessage
from twilio.rest import Client
from dotenv import load_dotenv
import threading

load_dotenv()

class AlertSystem:
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.smtp_email = os.getenv("SMTP_EMAIL", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        
        self.twilio_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.twilio_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.twilio_number = os.getenv("TWILIO_PHONE_NUMBER", "")
        self.target_number = os.getenv("TARGET_PHONE_NUMBER", "")
        
        self.twilio_client = None
        if self.twilio_sid and self.twilio_token:
            try:
                self.twilio_client = Client(self.twilio_sid, self.twilio_token)
            except Exception as e:
                print(f"Failed to initialize Twilio: {e}")

    def send_email_alert_sync(self, subject, body):
        if not self.smtp_email or not self.smtp_password:
            print("SMTP credentials not configured. Skipping email.")
            return
            
        try:
            msg = EmailMessage()
            msg.set_content(body)
            msg['Subject'] = subject
            msg['From'] = self.smtp_email
            msg['To'] = self.smtp_email # Send to self for alerts
            
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_email, self.smtp_password)
                server.send_message(msg)
            print(f"Email alert sent successfully: {subject}")
        except Exception as e:
            print(f"Failed to send email: {e}")

    def send_sms_alert_sync(self, body):
        if not self.twilio_client or not self.twilio_number or not self.target_number:
            print("Twilio credentials not fully configured. Skipping SMS.")
            return
            
        try:
            message = self.twilio_client.messages.create(
                body=body,
                from_=self.twilio_number,
                to=self.target_number
            )
            print(f"SMS alert sent successfully, SID: {message.sid}")
        except Exception as e:
            print(f"Failed to send SMS: {e}")

    def dispatch_alert(self, event_type, description, use_email=True, use_sms=False):
        subject = f"🚨 SECURITY ALERT: {event_type}"
        body = f"Smart Surveillance System detected an event:\n\nType: {event_type}\nDescription: {description}\n\nPlease check the dashboard or database for more details."
        
        if use_email:
            threading.Thread(target=self.send_email_alert_sync, args=(subject, body), daemon=True).start()
        
        if use_sms:
            threading.Thread(target=self.send_sms_alert_sync, args=(body,), daemon=True).start()
