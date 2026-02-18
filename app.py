import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Page config
st.set_page_config(page_title="Plant Health Detection", page_icon="🌿")

st.title("🌿 Plant Health Detection")
st.write("Upload a leaf image to check if it is Healthy or Unhealthy.")

# Load model (cached so it loads only once)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("healthy_unhealthy_model.h5")
    return model

model = load_model()

IMG_SIZE = 224

def predict_image(image):
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    p = model.predict(img)[0][0]

    if p >= 0.5:
        label = "Unhealthy"
        confidence = p * 100
    else:
        label = "Healthy"
        confidence = (1 - p) * 100

    return label, confidence

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, confidence = predict_image(image)

    if label == "Healthy":
        st.success(f"✅ Prediction: {label}")
    else:
        st.error(f"⚠️ Prediction: {label}")

    st.write(f"Confidence: {confidence:.2f}%")
