import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Plant Health Detection", page_icon="🌿")

st.title("🌿 Plant Health Detection")
st.write("Upload a **leaf image** to check if it is Healthy or Unhealthy.")
st.write("❌ Non-leaf images (dog, car, icon, etc.) will be rejected.")

# -------------------------------
# Load models (cached)
# -------------------------------
@st.cache_resource
def load_health_model():
    return tf.keras.models.load_model("healthy_unhealthy_model.h5")

@st.cache_resource
def load_validator_model():
    return tf.keras.applications.MobileNetV2(weights="imagenet")

health_model = load_health_model()
validator_model = load_validator_model()

IMG_SIZE = 224

# -------------------------------
# Safe preprocessing
# -------------------------------
def preprocess_for_health(image):
    # Force RGB (fix RGBA / grayscale)
    image = image.convert("RGB")

    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)
    return img

# -------------------------------
# Check if image is plant/leaf
# -------------------------------
def is_leaf_image(image):
    image = image.convert("RGB")
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = validator_model.predict(img)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]

    plant_keywords = [
        "plant", "leaf", "tree", "flower", "rose", "daisy", "sunflower",
        "corn", "maize", "wheat", "mushroom", "cucumber", "pumpkin",
        "strawberry", "pineapple", "banana", "jackfruit", "fig", "pepper"
    ]

    for _, label, prob in decoded:
        label_lower = label.lower()
        for kw in plant_keywords:
            if kw in label_lower:
                return True, decoded

    return False, decoded

# -------------------------------
# Predict health
# -------------------------------
def predict_health(image):
    img = preprocess_for_health(image)
    p = health_model.predict(img)[0][0]  # sigmoid output

    if p >= 0.5:
        label = "Unhealthy"
        confidence = p * 100
    else:
        label = "Healthy"
        confidence = (1 - p) * 100

    return label, confidence

# -------------------------------
# UI: File uploader
# -------------------------------
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("❌ Invalid image file. Please upload a valid image.")
        st.stop()

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Basic sanity check
    w, h = image.size
    if w < 50 or h < 50:
        st.error("❌ Image is too small. Please upload a real leaf photo.")
        st.stop()

    st.write("🔍 Checking if this is a leaf/plant image...")

    is_leaf, predictions = is_leaf_image(image)

    if not is_leaf:
        st.error("❌ This does NOT look like a leaf/plant image.")
        st.write("Top predictions from validator model:")
        for _, label, prob in predictions:
            st.write(f"- {label} : {prob*100:.2f}%")
        st.warning("Please upload a **leaf image** only.")
        st.stop()

    st.success("✅ Leaf/Plant detected. Running health analysis...")

    # Predict health
    label, confidence = predict_health(image)

    if label == "Healthy":
        st.success(f"🌿 Prediction: {label}")
    else:
        st.error(f"⚠️ Prediction: {label}")

    st.write(f"Confidence: {confidence:.2f}%")
