import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image  # FIX: missing import

# Load the TensorFlow model
# FIX: correct model path + safer loading
model = tf.keras.models.load_model('healthy_unhealthy_model.h5', compile=False)

def load_image(image_file):
    image = Image.open(image_file).convert("RGB")  # FIX: ensure RGB
    return image

def predict(image):
    image = np.array(image)  # FIX: convert PIL → NumPy

    image = cv2.resize(image, (224, 224))  # Resize
    image = image / 255.0  # FIX: normalization

    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)

    return predictions

# Streamlit UI
st.title('Plant Health Detection')
st.write('Upload an image of a plant leaf to check its health.')

# Image upload
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    predictions = predict(image)
    st.write('Prediction:', predictions)

# Webcam capture
if st.button('Capture from Webcam'):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # FIX: correct color
        st.image(frame_rgb, caption='Captured Image', use_container_width=True)

        predictions = predict(frame_rgb)
        st.write('Prediction:', predictions)

    cap.release()
