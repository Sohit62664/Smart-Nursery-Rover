import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Plant Health Detection", page_icon="🌿")

st.title("🌿 Plant Health Detection System")
st.write("Upload or capture an image to detect plant health (Healthy / Unhealthy).")

# =========================
# LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("healthy_unhealthy_model.h5")
    return model

model = load_model()

# =========================
# IMAGE VALIDATION
# =========================
def validate_image(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        # Blur detection using variance of Laplacian
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        return image_np, blur_score

    except Exception:
        return None, None

# =========================
# LEAF EXTRACTION (MULTI)
# =========================
def extract_leaves(image_np):
    leaves = []
    boxes = []

    # Convert to HSV
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    # Green color range (tuned for plants)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological operations (remove noise)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Filter small noise
        if area > 1500:
            x, y, w, h = cv2.boundingRect(cnt)

            leaf = image_np[y:y+h, x:x+w]
            leaves.append(leaf)
            boxes.append((x, y, w, h))

    return leaves, boxes

# =========================
# PREDICT SINGLE LEAF
# =========================
def predict_leaf(leaf_img):
    try:
        leaf = cv2.resize(leaf_img, (224, 224))
        leaf = leaf / 255.0
        leaf = np.expand_dims(leaf, axis=0)

        prediction = model.predict(leaf)[0][0]

        if prediction > 0.5:
            label = "Unhealthy"
            confidence = prediction
        else:
            label = "Healthy"
            confidence = 1 - prediction

        return label, float(confidence)

    except Exception:
        return "Error", 0.0

# =========================
# INPUT OPTIONS
# =========================
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("Capture Image")

image_bytes = None

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
elif camera_image is not None:
    image_bytes = camera_image.getvalue()

# =========================
# MAIN PROCESSING
# =========================
if image_bytes is not None:

    image_np, blur_score = validate_image(image_bytes)

    if image_np is None:
        st.error("❌ Invalid or corrupted image.")
    else:
        st.subheader("📷 Original Image")
        st.image(image_np, use_container_width=True)

        # Blur warning
        if blur_score < 100:
            st.warning("⚠️ Image seems blurry. Results may be inaccurate.")

        # Extract leaves
        leaves, boxes = extract_leaves(image_np)

        if len(leaves) == 0:
            st.error("❌ No leaf detected. Please upload a clear plant image.")
        else:
            st.subheader("🌿 Detected Leaves")

            results = []
            annotated_image = image_np.copy()

            for i, (leaf, box) in enumerate(zip(leaves, boxes)):
                x, y, w, h = box

                label, confidence = predict_leaf(leaf)
                results.append(label)

                # Draw bounding box
                color = (0, 255, 0) if label == "Healthy" else (255, 0, 0)
                cv2.rectangle(annotated_image, (x, y), (x+w, y+h), color, 2)

                cv2.putText(
                    annotated_image,
                    f"{i+1}: {label}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                # Show each leaf
                st.image(leaf, caption=f"Leaf {i+1}", width=200)

                st.write(f"Leaf {i+1} → {label} ({confidence*100:.2f}%)")

            # Show annotated image
            st.subheader("📍 Leaf Detection with Bounding Boxes")
            st.image(annotated_image, use_container_width=True)

            # =========================
            # FINAL DECISION
            # =========================
            if "Unhealthy" in results:
                st.error("🚨 Final Result: Plant Status → Unhealthy")
            else:
                st.success("✅ Final Result: Plant Status → Healthy")
