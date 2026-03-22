import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import cv2

# ===== Page Config =====
st.set_page_config(page_title="Plant Health Detection", page_icon="🌿")

st.title("🌿 Plant Health Detection")
st.write("Upload or capture a plant image to detect and analyze each leaf.")

# ===== Load Model =====
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("healthy_unhealthy_model.h5")

model = load_model()

IMG_SIZE = 224

# ===== MULTIPLE LEAF EXTRACTION =====
def extract_multiple_leaves(image):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Improved green range (handles light/dark leaves)
    lower = np.array([20, 30, 30])
    upper = np.array([100, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Edge detection to separate touching leaves
    edges = cv2.Canny(mask, 50, 150)

    # Combine mask + edges
    combined = cv2.bitwise_or(mask, edges)

    # Clean noise
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    leaves = []

    for c in contours:
        if cv2.contourArea(c) < 300:
            continue

        x, y, w, h = cv2.boundingRect(c)

        leaf = img[y:y+h, x:x+w]
        leaf = cv2.cvtColor(leaf, cv2.COLOR_BGR2RGB)

        leaves.append(Image.fromarray(leaf))

    return leaves


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


# ===== INPUT OPTION =====
option = st.radio("Choose Input Method:", ["Upload Image", "Use Camera"])

image = None

# ===== Upload =====
if option == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(BytesIO(uploaded_file.read())).convert("RGB")

# ===== Camera =====
elif option == "Use Camera":
    camera_file = st.camera_input("📷 Take a photo")

    if camera_file is not None:
        image = Image.open(BytesIO(camera_file.read())).convert("RGB")

# ===== PROCESS =====
if image is not None:
    try:
        st.image(image, caption="Original Image", use_column_width=True)

        leaves = extract_multiple_leaves(image)

        if len(leaves) == 0:
            st.warning("⚠️ No leaf detected. Try a clearer image.")
        else:
            st.success(f"🌿 Detected {len(leaves)} leaf/leaves")

            for i, leaf in enumerate(leaves):
                st.image(leaf, caption=f"Leaf {i+1}", use_column_width=True)

                label, confidence = predict_image(leaf)

                if label == "Healthy":
                    st.success(f"Leaf {i+1}: Healthy ({confidence:.2f}%)")
                else:
                    st.error(f"Leaf {i+1}: Unhealthy ({confidence:.2f}%)")

    except Exception as e:
        st.error(f"❌ Error: {e}")
