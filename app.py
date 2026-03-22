import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import cv2

# ===== Page Config =====
st.set_page_config(page_title="Plant Health Detection", page_icon="🌿")

st.title("🌿 Plant Health Detection")
st.write("Upload or capture a leaf image to check if it is Healthy or Unhealthy.")

# ===== Load Model =====
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("healthy_unhealthy_model.h5")
    return model

model = load_model()

IMG_SIZE = 224

# ===== MULTI-LEAF DETECTION =====
def detect_leaves(image):
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([20, 30, 30])
    upper = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    leaves = []
    boxes = []

    for c in contours:
        if cv2.contourArea(c) < 1000:
            continue

        x, y, w, h = cv2.boundingRect(c)
        leaf = img[y:y+h, x:x+w]
        leaf = cv2.cvtColor(leaf, cv2.COLOR_BGR2RGB)

        leaves.append(Image.fromarray(leaf))
        boxes.append((x, y, w, h))

    return leaves, boxes

# ===== PREDICTION =====
def predict_image(image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(image).astype(np.float32)

    # IMPORTANT: match training preprocessing
    img = img / 255.0   # keep if used during training

    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)

    # Handle both cases: sigmoid or softmax
    if pred.shape[-1] == 1:
        # Binary sigmoid
        prob = pred[0][0]
        return prob
    else:
        # Softmax
        prob = pred[0]
        return prob

# ===== INPUT OPTION =====
option = st.radio("Choose Input Method:", ["Upload Image", "Use Camera"])

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(BytesIO(uploaded_file.read())).convert("RGB")

else:
    camera_file = st.camera_input("📷 Take a photo")
    if camera_file:
        image = Image.open(BytesIO(camera_file.read())).convert("RGB")

# ===== PROCESS =====
if image is not None:
    try:
        st.image(image, caption="Original Image", use_column_width=True)

        leaves, boxes = detect_leaves(image)

        if len(leaves) == 0:
            st.warning("⚠️ No leaf detected. Using full image.")
            leaves = [image]
            boxes = [(0, 0, image.size[0], image.size[1])]

        result_img = np.array(image)

        st.subheader("🔍 Leaf-wise Prediction")

        for i, (leaf, box) in enumerate(zip(leaves, boxes)):

            pred = predict_image(leaf)

            # ===== FIXED LABEL LOGIC =====
            if isinstance(pred, np.ndarray):
                # Softmax case
                class_idx = np.argmax(pred)

                if class_idx == 0:
                    label = "Healthy"
                    confidence = pred[0] * 100
                    color = (0, 255, 0)
                else:
                    label = "Unhealthy"
                    confidence = pred[1] * 100
                    color = (0, 0, 255)

                raw_value = pred

            else:
                # Sigmoid case
                prob = pred

                # 🔥 Try flipping if wrong (common issue)
                if prob > 0.5:
                    label = "Unhealthy"
                    confidence = prob * 100
                    color = (0, 0, 255)
                else:
                    label = "Healthy"
                    confidence = (1 - prob) * 100
                    color = (0, 255, 0)

                raw_value = prob

            st.image(leaf, caption=f"Leaf {i+1}")
            st.write(f"Prediction: {label}")
            st.write(f"Confidence: {confidence:.2f}%")
            st.write(f"Raw Output: {raw_value}")

            x, y, w, h = box
            cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(result_img, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        st.subheader("📦 Final Output")
        st.image(result_img, caption="Detected Leaves", use_column_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")
