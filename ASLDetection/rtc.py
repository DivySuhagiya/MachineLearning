import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
from googletrans import Translator
from gtts import gTTS
import os

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

# Label dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
               15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}

st.title("🖐️ ASL Hand Sign Detection & Translation")

# Initialize session state for camera and sentence tracking
if "detected_sentence" not in st.session_state:
    st.session_state.detected_sentence = ""  
if "last_detection_time" not in st.session_state:
    st.session_state.last_detection_time = time.time()
if "last_detected_letter" not in st.session_state:
    st.session_state.last_detected_letter = None
if "translated_sentence" not in st.session_state:
    st.session_state.translated_sentence = "" 

# Translator
translator = Translator()

# Available Languages
language_dict = {
    "Arabic": "ar",
    "Chinese (Simplified)": "zh-cn",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Gujarati": "gu",
    "Hindi": "hi",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Spanish": "es",
}

# Layout for Buttons
col1, col2, col3 = st.columns(3)

with col1:
    start_camera = st.button('Start Camera')

with col2:
    stop_camera = st.button('Stop Camera')

with col3:
    clear_sentence = st.button('Clear Sentence')

# Clear sentence 
if clear_sentence:
    st.session_state.detected_sentence = ""
    st.session_state.translated_sentence = ""

# Display detected sentence always
sentence_placeholder = st.empty()
sentence_placeholder.write(
    f"**Detected Sentence:** {st.session_state.detected_sentence}")

# Language selection dropdown
selected_language = st.selectbox(
    "🌍 Select Language for Translation", list(language_dict.keys()))

# Translate button
if st.button("Translate"):
    sentence = st.session_state.detected_sentence.strip()
    if st.session_state.detected_sentence.strip():
        translated_text = translator.translate(
            sentence, dest=language_dict[selected_language])
        st.session_state.translated_sentence = translated_text.text

        # Convert Translated Text to Speech
        tts = gTTS(st.session_state.translated_sentence,
                   lang=language_dict[selected_language])
        tts.save("translated_audio.mp3")

        # Play the Speech in Streamlit
        st.audio("translated_audio.mp3", format="audio/mp3")
    else:
        st.session_state.translated_sentence = "No text to translate."

# Display translated sentence
translated_placeholder = st.empty()
translated_placeholder.write(
    f"**Translated Sentence ({selected_language}):** {st.session_state.translated_sentence}")

# Video Capture 
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_window = st.image([])

while start_camera and not stop_camera:
    data_aux = []
    x_, y_ = [], []

    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to access the webcam!")
        break

    # frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            data_aux = [(lm.x - min(x_), lm.y - min(y_))
                        for lm in hand_landmarks.landmark]
            # Flatten list
            data_aux_flat = [coord for pair in data_aux for coord in pair]

        # Bounding Box for Hand
        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
        x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10

        # Predict ASL Sign
        prediction = model.predict([np.asarray(data_aux_flat)])
        predicted_character = labels_dict[int(prediction[0])]

        # Add delay to avoid rapid letter detection
        if predicted_character != st.session_state.last_detected_letter:
            st.session_state.last_detection_time = time.time()
            st.session_state.last_detected_letter = predicted_character

        elif time.time() - st.session_state.last_detection_time > 2:  # Add letter after 2 seconds
            if predicted_character == "del":
                st.session_state.detected_sentence = st.session_state.detected_sentence[:-1]  # Remove last character
            elif predicted_character == "space":
                st.session_state.detected_sentence += " "  # Add space
            else:
                 st.session_state.detected_sentence += predicted_character
            st.session_state.last_detection_time = time.time()
            sentence_placeholder.write(
                f"**Detected Sentence:** {st.session_state.detected_sentence}")

        # Draw Bounding Box & Label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Display Processed Frame
    frame_window.image(frame, channels="BGR")

    # # Stop camera when button is clicked
    # if stop_camera:
    #     st.session_state.camera_running = False
    #     break

# Release camera when stopping
if cap:
    cap.release()
    cv2.destroyAllWindows()
