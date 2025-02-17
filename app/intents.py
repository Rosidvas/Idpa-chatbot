import nltk
from gmail import create_credentials, getMessages, getUnreadMessages, getMessage, readMessage, readOpenMessage, updateMessageRead

credentials = None
messageList = None
unreadMessageList = None
currentEmail = None

#creates credentials to use the Gmail API
def createCredentials():
    global credentials
    creds = create_credentials()
    credentials = creds

#gets list of email
def getEmails(tickUnread, creds, lang):
    global messageList
    global unreadMessageList

    if tickUnread is True:
        emails = getMessages(creds=creds)
        messageList = emails
        return messageList #switched emails variable to this
    else:
        emails = getUnreadMessages(creds=creds)
        unreadMessageList = emails
        return unreadMessageList #switched emails variable to this

#get a specific email by ID
def selectEmail(tickUnread, input, credentials):
    global messageList
    global currentEmail

    mailNum = [int(num)for num in input.split() if num.isdigit()] #converts to int
    mailNum[0] = mailNum[0] - 1
    displayNum = mailNum[0]
    print(displayNum)

    if tickUnread is True:
        email = getMessage(unreadMessageList, credentials, mailNum[0])       
    else:
        email = getMessage(messageList, credentials, mailNum[0])
        
    currentEmail = email
    return email

#reads and sends current Email contents
def readOpenEmail(msg, credentials): 
    sender, subject, snippet = readOpenMessage(msg)
    data = [sender, subject, snippet]
    return data

#Marks current email as read: regardless if already read or not
def markEmailRead(msg, credentials):
    updateMessageRead(msg, credentials)
    msgId = msg['id']
    for email in unreadMessageList:
        if msgId in email:
            unreadMessageList.remove(email)
            print("message removed from list")          
    return

#Handles different task Intents: Email handling, ...
def intent_classifier(input, intent, lang):
    response = ["","",""]
    try:
        match intent:
            case 'createCreds':
                createCredentials()
                return response
            case 'getEmailsList':         
                msgRaw = getEmails(True, credentials, lang)
                return response
            case 'getUnreadEmailsList':
                msgUnreadRaw = getEmails(False, credentials, lang)
                return response
            case 'countEmails':
                response = [len(messageList), "", ""]
                return response
            case 'countUnreadEmails':
                response = [len(unreadMessageList), "", ""]
                return response
            case 'selectEmail':
                email = selectEmail(False, input, credentials)
                return response
            case 'selectUnreadEmail':
                email = selectEmail(True, input, credentials)
                return response
            case 'readEmail':
                readMessage(currentEmail)
                data = readOpenMessage(currentEmail)
                return data #returns array in order -> Sender -> Subject -> Snippet
            case 'markEmailAsRead':
                try:
                    markEmailRead(currentEmail, credentials)
                    return response
                except Exception as error:
                    print(f'An error occured {error}')
                    return False
            case _:
                print('Intent does not match with any action')
                return response
    except Exception as e:
        print(f"An error occured during task handling: {e}")
        return False


#Functionality of Getting List of Read/Unread Emails
def test_1():
    creds, response = intent_classifier('Create credentials', 'createCreds', 'english') #structure -> 
    print(response)

    list = intent_classifier('Get all emails', 'getEmailsList', 'english')
    print(f'Email List: {len(list)}')

    list = intent_classifier('Get all unread Emails', 'getUnreadEmailsList', 'english')
    print(f'unread Email List: {len(list)}')

#Functionality Email Selection and Reading
def test_2():
    creds, response = intent_classifier('Create credentials', 'createCreds', 'english')
    print(response)

    list = intent_classifier('Get all unread Emails', 'getUnreadEmailsList', 'english')
    Email = intent_classifier('Select Email 25', 'selectEmail', 'english')
    data = intent_classifier('Read currently selected Email', 'readEmail', 'english')
    print(f"Emails: {len(list)}, current Email: {Email}, data length: {len(data)}")

#Functionality of Language Switching and Marking Emails that have been read
def test_3():
    try:

        creds, response = intent_classifier('Create some credentials', 'createCreds', 'english')

        #1st Iteration, Email not read
        list = intent_classifier('Get all unread Emails', 'getUnreadEmailsList', 'english') 
        Email = intent_classifier('Select Email 0', 'selectEmail', 'english')
        data = intent_classifier('Read the message', 'readEmail', 'english')
        intent_classifier('Mark current message as read', 'markEmailAsRead', 'english')

        #2nd Iteration, Email marked read
        list = intent_classifier('Get all unread Emails', 'getUnreadEmailsList', 'english') 
        Email = intent_classifier('Select Email 0', 'selectEmail', 'english')
        data = intent_classifier('Read the new Message', 'readEmail', 'english')
        print('test success')
    except Exception as e:
        print(f"Test unsuccessful! error: {e}")

def test_4():
    try:
        print("")
    except Exception as e:
        print("d")
    return




