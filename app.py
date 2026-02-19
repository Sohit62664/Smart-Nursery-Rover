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
st.write("⚠️ If you upload a non-leaf image (dog, human, car, etc.), it will be rejected.")

# -------------------------------
# Load models (cached)
# -------------------------------
@st.cache_resource
def load_health_model():
    model = tf.keras.models.load_model("healthy_unhealthy_model.h5")
    return model

@st.cache_resource
def load_validator_model():
    # Pretrained ImageNet model to check if image is a plant/leaf
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model

health_model = load_health_model()
validator_model = load_validator_model()

IMG_SIZE = 224

# -------------------------------
# Utility: Preprocess image
# -------------------------------
def preprocess_image(image, size=224):
    img = np.array(image)

    # If grayscale, convert to 3 channels
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.resize(img, (size, size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------------
# Step 1: Check if image is a plant/leaf
# -------------------------------
def is_leaf_image(image):
    # Preprocess for MobileNetV2 (expects 224x224 and specific scaling)
    img = np.array(image)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.resize(img, (224, 224))
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    preds = validator_model.predict(img)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]

    # Keywords that indicate plant/leaf/flower/tree
    plant_keywords = [
        "plant", "leaf", "tree", "flower", "rose", "daisy", "sunflower",
        "corn", "maize", "wheat", "mushroom", "cucumber", "pumpkin",
        "strawberry", "pineapple", "banana", "jackfruit", "fig"
    ]

    # Check top-5 predictions
    for _, label, prob in decoded:
        label_lower = label.lower()
        for kw in plant_keywords:
            if kw in label_lower:
                return True, decoded

    return False, decoded

# -------------------------------
# Step 2: Predict health
# -------------------------------
def predict_health(image):
    img = preprocess_image(image, IMG_SIZE)
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
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("🔍 Checking if this is a leaf/plant image...")

    is_leaf, predictions = is_leaf_image(image)

    if not is_leaf:
        st.error("❌ This does NOT look like a leaf/plant image.")
        st.write("Top predictions from validator model:")
        for _, label, prob in predictions:
            st.write(f"- {label} : {prob*100:.2f}%")
        st.warning("Please upload a **leaf image** only.")
    else:
        st.success("✅ Leaf/Plant detected. Running health analysis...")

        label, confidence = predict_health(image)

        if label == "Healthy":
            st.success(f"🌿 Prediction: {label}")
        else:
            st.error(f"⚠️ Prediction: {label}")

        st.write(f"Confidence: {confidence:.2f}%")
