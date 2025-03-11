🖐️ ASL Hand Sign Detection & Translation

This project detects American Sign Language (ASL) hand gestures using a trained machine learning model and translates them into text and speech in multiple languages.

🎥 [Demo Video](https://youtu.be/AtUeDyo1ikU?si=sGkAX2vag0f9b4iY)

✨ Features

Real-time hand gesture recognition using MediaPipe.
Predicts ASL signs and constructs a sentence.
Translates the detected sentence into different languages using Google Translate.
Converts translated text to speech with gTTS (Google Text-to-Speech).
Streamlit-based UI for a simple and interactive experience.

🚀 Usage

Run the Streamlit app:
streamlit run app.py

Use the UI to:
Start the camera.
Detect hand gestures and form sentences.
Translate and play audio for the detected sentence.

🛠️ Technologies Used

Python
OpenCV - for real-time camera input processing.
MediaPipe - for hand landmark detection.
Streamlit - for an interactive UI.
Scikit-learn - for machine learning model handling.
Google Translate API - for text translation.
gTTS (Google Text-to-Speech) - for converting translated text to speech.

🏗 Model Training

The model is trained using Scikit-learn on a dataset of ASL hand signs, processed with MediaPipe landmarks. It classifies gestures into 29 categories (A-Z, space, delete, nothing).

🖼 Example Output

Example of ASL hand gesture recognition.

🔍 Future Enhancements

Support for real-time sentence suggestions.
Improved model accuracy with a larger dataset.
Mobile version using TensorFlow Lite.



