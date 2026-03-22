import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image  # ✅ FIX 1: Missing import

# Load the TensorFlow model
model = tf.keras.models.load_model('path/to/your/model')

def load_image(image_file):
   image = Image.open(image_file).convert("RGB")  # ✅ ensure RGB
   return image

def predict(image):
    image = np.array(image)  # ✅ FIX 2: Convert PIL → NumPy

    image = cv2.resize(image, (224, 224))  # Resize image
    image = image / 255.0  # ✅ FIX 3: Normalize

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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # ✅ FIX 4: Color correction
        st.image(frame_rgb, caption='Captured Image', use_container_width=True)

        predictions = predict(frame_rgb)
        st.write('Prediction:', predictions)

    cap.release()
