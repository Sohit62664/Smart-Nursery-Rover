import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

# ===== Page Config =====
st.set_page_config(page_title="Plant Health Detection", page_icon="🌿")

st.title("🌿 Plant Health Detection")
st.write("Upload a leaf image to check if it is Healthy or Unhealthy.")

# ===== Load Model =====
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("healthy_unhealthy_model.h5")

model = load_model()

IMG_SIZE = 224

# ===== Prediction Function =====
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

# ===== Upload =====
uploaded_file = st.file_uploader(
    "📤 Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        # 🔥 FIXED IMAGE LOADING
        image = Image.open(BytesIO(uploaded_file.read())).convert("RGB")

        st.image(image, caption="Uploaded Image", use_column_width=True)

        label, confidence = predict_image(image)

        if label == "Healthy":
            st.success(f"✅ Prediction: {label}")
        else:
            st.error(f"⚠️ Prediction: {label}")

        st.write(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
