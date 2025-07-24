import webbrowser
import joblib
import speech_recognition as sr
from datetime import datetime

# ✅ Load the trained intent classification model
model = joblib.load("intent_classifier.pkl")

# ✅ Define function to handle predicted intents
def handle_intent(intent):
    if intent in ["open_youtube"]:
        print("🔗 Opening YouTube...")
        webbrowser.open("https://www.youtube.com")
    elif intent == "get_time" or intent == "tell_time":
        current_time = datetime.now().strftime("%I:%M %p")
        print(f"⏰ The current time is: {current_time}")
    elif intent == "open_camera":
        print("📸 Opening camera (simulated)...")
    else:
        print("🤖 Sorry, I didn’t understand that command.")


# ✅ Start listening to user input
r = sr.Recognizer()

with sr.Microphone() as source:
    print("🎤 Adjusting for ambient noise...")
    r.adjust_for_ambient_noise(source, duration=1)

    print("🗣️ Speak now...")
    audio = r.listen(source)

try:
    command = r.recognize_google(audio)
    print(f"✅ You said: {command}")

    # ✅ Predict the intent
    intent = model.predict([command])[0]
    print(f"🎯 Predicted intent: {intent}")

    # ✅ Handle the intent
    handle_intent(intent)

except sr.UnknownValueError:
    print("❌ Could not understand audio. Please speak clearly.")
except sr.RequestError as e:
    print(f"⚠️ Could not request results from Google Speech Recognition service; {e}")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
