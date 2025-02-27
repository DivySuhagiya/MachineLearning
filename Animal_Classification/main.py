import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

class_labels = {
    0: "Bear", 1: "Bird", 2: "Cat", 3: "Cow", 4: "Deer", 5: "Dog",
    6: "Dolphin", 7: "Elephant", 8: "Giraffe", 9: "Horse", 10: "Kangaroo",
    11: "Lion", 12: "Panda", 13: "Tiger", 14: "Zebra"
}

Max_File_Size = 5 * 1024 * 1024

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        "InceptionV3_Train.h5")
    return model


model = load_model()

# Function to preprocess the image
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize
    image = np.reshape(image, (1, 224, 224, 3))
    return image


# Streamlit UI
st.title("Animal Classification")
st.write("Model can predict: Bear, Bird, Cat, Cow, Deer, Dog, Dolphin, Elephant, Giraffe, Horse, Kangaroo, Lion, Panda, Tiger, Zebra")

uploaded_file = st.file_uploader("Upload an image", type=[
                                 "jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_file is not None:
    if uploaded_file.size > Max_File_Size:
        st.error("File size is too large. Please upload a file smaller than 5MB.")
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    predicted_class = class_labels[predicted_label]

    # Display prediction results
    st.write(
        f"### The model is {confidence:.2f}% confident that this is a {predicted_class}.")
