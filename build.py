import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle

# ✅ Corrected Text Samples
texts = [
    "open camera", "start the camera", "take a photo", "camera kholo", "photo lena hai",
    "open youtube", "start youtube", "youtube kholo", "play video", "mujhe youtube chahiye",
    "what time is it", "tell me the time", "current time", "kitna time hua", "kya time hua",
    "whatsapp kholo", "whatsapp open karo", "whatsapp khol do", "whatsapp chalu karo", "whatsapp kaam karo",
    "open facebook", "facebook kholo", "facebook open karo", "facebook chalu karo", "facebook kaam karo",
    "open instagram", "instagram kholo", "instagram open karo", "instagram chalu karo", "instagram kaam karo",
    "open clock", "clock kholo", "clock open karo", "clock chalu karo", "clock kaam karo"
]

# ✅ Matching Labels (35 items)
labels = [ 
    "open_camera", "open_camera", "open_camera", "open_camera", "open_camera",
    "open_youtube", "open_youtube", "open_youtube", "open_youtube", "open_youtube",
    "tell_time", "tell_time", "tell_time", "tell_time", "tell_time",
    "open_whatsapp", "open_whatsapp", "open_whatsapp", "open_whatsapp", "open_whatsapp",
    "open_facebook", "open_facebook", "open_facebook", "open_facebook", "open_facebook",
    "open_instagram", "open_instagram", "open_instagram", "open_instagram", "open_instagram",
    "open_clock", "open_clock", "open_clock", "open_clock", "open_clock"
]

# ✅ Encode Labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# ✅ Tokenize Inputs
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=5)

# ✅ One-hot encode labels
y_cat = to_categorical(y, num_classes=num_classes)

# ✅ Build Model
model = Sequential([
    Embedding(input_dim=100, output_dim=16, input_length=5),
    LSTM(32),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_cat, epochs=200, verbose=0)


# ✅ Save model
model.save("intent_model.h5")
print("✅ Model trained and saved as intent_model.h5")

# ✅ Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer saved as tokenizer.pkl")

# ✅ Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("✅ LabelEncoder saved as label_encoder.pkl")
