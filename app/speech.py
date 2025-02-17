import pyaudio
import wave
import threading
import pyttsx3
import speech_recognition as sr
from chatbot import analyze_user_input, set_language

file_path = "./recording/recorded_input.wav"
FORMAT = pyaudio.paInt16  
CHANNELS = 1             
RATE = 44100             
CHUNK = 1024             


p = pyaudio.PyAudio()
frames = []
recording = False
stream = None
input_string = None
voice_number = 1

#Changes voice language based on user's keyboard input
def switch_voice_lang(num):
    global voice_number
    voice_number = num

#starts recording user audio
def start_recording():
    global recording, stream
    if not recording:
        recording = True
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, stream_callback=callback)
        print("Recording started.")

#Stops recording user audio
def stop_recording():
    global recording, stream
    if recording:
        recording = False
        if stream is not None:
            stream.stop_stream()
            stream.close()
            print("Recording stopped.")
            save_recording()
        else:
            print("No active recording to stop.")
    else:
        print("No recording in progress.")

#Saves recorded audio
def save_recording():
    global frames, file_path
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    print("Recording saved in", file_path) # If there is a previous Audio recording, it will be overwritten
    convert_speech_text()
    frames.clear()

# Converts speech to text
def convert_speech_text():
    global file_path
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio, language="de-DE") # de-DE fr-FR
        print("Text: " + text)
        model, lang = set_language(text) # OPTIONAL || WILL REMOVE LATER
        response = analyze_user_input("user", text, model, lang) #
        convert_text_speech(response)
    except Exception as e:
        print("Exception: " + str(e))


#Converts response to speech
def convert_text_speech(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[voice_number].id)
    
    try:
        engine.say(text)
        print(text)
        engine.runAndWait()
        engine.stop()
        
    except Exception as e:
        print("Exception occurred: ", e)

def callback(in_data):
    global recording
    if recording:
        frames.append(in_data)
        return (in_data, pyaudio.paContinue)
    else:
        return (b'', pyaudio.paComplete)