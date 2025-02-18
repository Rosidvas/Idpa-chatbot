import os.path
import email
import base64
import json
import re
import time
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging
import requests

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.modify'] #,
creds = None

#Create Credentials for using the Gmail-API
def create_credentials():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(               
                # your creds file here.
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

# GET list of Emails read/unread
def getMessages(creds):
    emails_list = []
    emailData_list = []
    try:
        query = 'in:inbox newer_than:7d -category:{social promotions updates forums}'
        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(userId = 'me', labelIds=['INBOX'], q=query).execute()
        messages = results.get("messages", [])
        if not messages:
            print("no meesages found")
        else:
            for message in messages:
                emails_list.append(message['id'])
                emailData_list.append(message)
            return emails_list
    except Exception as Error:
        print(f"An error occured: {Error}")

#GET list of Emails unread
def getUnreadMessages(creds):
    unread_emails_list = []
    query='in:inbox is:unread newer_than:7d -category:{social promotions updates forums}' #
    try: 
        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(userId='me', labelIds=['UNREAD'], q=query).execute() #, q=["is:unread"] ", q=["is:unread"] in:inbox -category:{social promotions updates forums}"
        messages = results.get("messages", [])
        if not messages:
            print("all messages read")
        else:
            for message in messages:
                unread_emails_list.append(message['id'])
        return unread_emails_list
    except Exception as error:
        print(f'An error occurred: {error}')

#SELECT an Email
def getMessage(list, creds, number):
    try:
        service = build('gmail', 'v1', credentials=creds)
        result = service.users().messages().get(userId='me', id=list[number], format='full').execute()
        return result
    except Exception as error:
        print(f"An error has occured: {error}")

#UPDATES current message as read
def updateMessageRead(msg, creds):
    msg = msg['id']

    try:
        service = build('gmail', 'v1', credentials=creds)
        service.users().messages().modify(userId='me', id=msg, body={'removeLabelIds': ['UNREAD']}).execute()
        print(f'marked Email as read')
    except Exception as error:
        print(f'An error occured: {error}')   

#Opens The Message and reads the entire content
def readMessage(msg):
    header = msg["payload"]["headers"]
    sender = [i['value'] for i in header if i['name']=="From"]
    subject = [i['value'] for i in header if i['name']=='Subject']
    snippet = msg['snippet']

    print(f'Message from: {sender}, subject: {subject}')
    print(f'Message snippet: {snippet}') #%s' % msg['snippet']
    return

#Opens Message but sends content instead of reading
def readOpenMessage(msg):
    header = msg["payload"]["headers"]
    sender = [i['value'] for i in header if i['name']=="From"]
    subject = [i['value'] for i in header if i['name']=='Subject']
    snippet = msg['snippet']

    return subject, sender, snippet

#Testing Email Functions
def test():
    creds = create_credentials()
    
    list = getMessages(creds)
    unread_list = getUnreadMessages(creds)
    print(f"Unread Message count: {len(unread_list)}")
    print(f"All Message count: {len(list)}")
    
    for email in range(len(unread_list)):
        message = getMessage(unread_list, creds, email)
        readMessage(message)
    
#test()

# Acquire all unread %
# Acquire a full list %
# read a message from list %
# mark unread message as read %
# alert of new notification


