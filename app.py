import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import cv2

# ===== Page Config =====
st.set_page_config(page_title="Leaf Detection", page_icon="🌿")

st.title("🌿 Leaf Detection Only")
st.write("Upload or capture an image to detect and extract leaves.")

# ===== LEAF DETECTION FUNCTION =====
def detect_leaves(image):
    img = np.array(image)
    original = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Green color range (adjustable)
    lower = np.array([20, 30, 30])
    upper = np.array([100, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Edge detection (helps separate leaves)
    edges = cv2.Canny(mask, 50, 150)

    # Combine mask + edges
    combined = cv2.bitwise_or(mask, edges)

    # Remove noise
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    leaves = []
    boxed_img = img.copy()

    for c in contours:
        if cv2.contourArea(c) < 300:
            continue

        x, y, w, h = cv2.boundingRect(c)

        # Draw bounding box
        cv2.rectangle(boxed_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Crop leaf
        leaf = img[y:y+h, x:x+w]
        leaf = cv2.cvtColor(leaf, cv2.COLOR_BGR2RGB)

        leaves.append(Image.fromarray(leaf))

    # Convert boxed image back to RGB
    boxed_img = cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB)

    return Image.fromarray(boxed_img), leaves


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

        boxed_image, leaves = detect_leaves(image)

        st.image(boxed_image, caption="Detected Leaves (Bounding Boxes)", use_column_width=True)

        if len(leaves) == 0:
            st.warning("⚠️ No leaves detected")
        else:
            st.success(f"🌿 Detected {len(leaves)} leaf/leaves")

            for i, leaf in enumerate(leaves):
                st.image(leaf, caption=f"Leaf {i+1}", use_column_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")
