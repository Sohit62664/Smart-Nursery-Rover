import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import cv2

# ===== Page Config =====
st.set_page_config(page_title="Plant Health Detection", page_icon="🌿")

st.title("🌿 Plant Health Detection")
st.write("Upload or capture a real leaf image to check if it is Healthy or Unhealthy.")

# ===== Load Model =====
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("healthy_unhealthy_model.h5")

model = load_model()

IMG_SIZE = 224

# ===== Leaf Extraction =====
def extract_leaf(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Green mask (can tune)
    lower = np.array([25, 40, 40])
    upper = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return image  # fallback

    # Largest contour
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    leaf = img[y:y+h, x:x+w]
    leaf = cv2.cvtColor(leaf, cv2.COLOR_BGR2RGB)

    return Image.fromarray(leaf)

# ===== Image Quality Check =====
def is_valid_leaf_image(image):
    img = np.array(image)

    # Check variance (low = blur/screen)
    variance = np.var(img)

    if variance < 500:
        return False

    return True

# ===== Prediction =====
def predict_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    p = model.predict(img)[0][0]

    if p >= 0.5:
        return "Unhealthy", p * 100
    else:
        return "Healthy", (1 - p) * 100

# ===== Input Option =====
option = st.radio("Choose Input Method:", ["Upload Image", "Use Camera"])

image = None

# ===== Upload =====
if option == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(BytesIO(uploaded_file.read())).convert("RGB")

# ===== Camera =====
elif option == "Use Camera":
    camera_file = st.camera_input("📷 Take a photo")
    if camera_file is not None:
        image = Image.open(BytesIO(camera_file.read())).convert("RGB")

# ===== Processing =====
if image is not None:
    try:
        st.image(image, caption="Original Image", use_column_width=True)

        # Leaf extraction
        leaf = extract_leaf(image)
        st.image(leaf, caption="Detected Leaf", use_column_width=True)

        # Quality check
        if not is_valid_leaf_image(leaf):
            st.warning("⚠️ Please capture a clear REAL leaf image (avoid screens / blur)")
        else:
            # Prediction
            label, confidence = predict_image(leaf)

            if label == "Healthy":
                st.success(f"✅ Prediction: {label}")
            else:
                st.error(f"⚠️ Prediction: {label}")

            st.write(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
