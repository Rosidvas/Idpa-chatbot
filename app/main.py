import keyboard
from chatbot import set_language, analyze_user_input
from speech import start_recording, stop_recording, switch_voice_lang

voice_chat_engaged = False

def onVoiceChat():
    print("voice mode triggered")
    while True:
        if keyboard.is_pressed('1'):
            start_recording()
        elif keyboard.is_pressed('2'):
            stop_recording()
        elif keyboard.is_pressed('3'):
            onTextChat()

        #Switches voice language based on key input
        #Once a speech-language classification algorithm can be implemented, this block will be removed
        if keyboard.is_pressed('f'):
            switch_voice_lang(4)
        elif keyboard.is_pressed('d'):
            switch_voice_lang(2)
        elif keyboard.is_pressed('e'):
            switch_voice_lang(1)
            
def onTextChat():
    print("text mode triggered")
    username = "Developer"
    while True:
        user_input = input("user: ")
        if "voicemode" in user_input:
            onVoiceChat()
        model, lang = set_language(user_input)
        response = analyze_user_input(username, user_input, model, lang)
        print(f"model: {response}")
        
while True:
    if keyboard.is_pressed('v'):
        onVoiceChat()
    elif keyboard.is_pressed('t'):
        onTextChat()