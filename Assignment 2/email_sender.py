def send_email(**kwargs):
    import smtplib
    import ssl
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    new_entries = kwargs["ti"].xcom_pull(task_ids="check_new_records")
    if not new_entries:
        return "No new records found."

    sender_email = ""  # Enter email here
    receiver_email = ""  # Enter email here
    password = ""  # Enter password here

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "New News Articles Available"

    body = f"New articles detected: {new_entries} new entries."
    message.attach(MIMEText(body, "plain"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())
        server.quit()

    return "Email sent successfully"
