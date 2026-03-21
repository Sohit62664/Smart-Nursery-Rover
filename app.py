import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ===== Page Config =====
st.set_page_config(page_title="Plant Health Detection", page_icon="🌿")

st.title("🌿 Plant Health Detection")
st.write("Upload a leaf image to check if it is Healthy or Unhealthy.")

# ===== Load Model (cached) =====
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("healthy_unhealthy_model.h5")
    return model

model = load_model()

IMG_SIZE = 224

# ===== Prediction Function =====
def predict_image(image):
    # Ensure image is RGB (handles RGBA, grayscale, etc.)
    image = image.convert("RGB")

    # Resize image safely
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to numpy array
    img = np.array(image)

    # Normalize pixel values
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Model prediction
    prediction = model.predict(img)

    # Extract probability
    p = prediction[0][0]

    if p >= 0.5:
        label = "Unhealthy"
        confidence = p * 100
    else:
        label = "Healthy"
        confidence = (1 - p) * 100

    return label, confidence


# ===== File Upload =====
uploaded_file = st.file_uploader(
    "📤 Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# ===== If Image Uploaded =====
if uploaded_file is not None:
    try:
        # Open image
        image = Image.open(uploaded_file).convert("RGB")

        # Display image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Predict
        label, confidence = predict_image(image)

        # Show result
        if label == "Healthy":
            st.success(f"✅ Prediction: {label}")
        else:
            st.error(f"⚠️ Prediction: {label}")

        st.write(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
