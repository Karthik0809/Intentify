import speech_recognition as sr
import pyttsx3
import threading

# Initialize recognizer and TTS engine
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()
speak_lock = threading.Lock()  # Lock for safe access to TTS

def listen():
    with sr.Microphone() as source:
        print("🎤 Listening...")
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, I didn't catch that."
    except sr.RequestError:
        return "Speech service unavailable."

def speak(text):
    with speak_lock:
        tts_engine.stop()  # Stop any ongoing speech
        tts_engine.say(text)
        tts_engine.runAndWait()
