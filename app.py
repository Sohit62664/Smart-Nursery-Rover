import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image  # FIX: missing import

# Load the TensorFlow model
# FIX: correct model path + safer loading
model = tf.keras.models.load_model('healthy_unhealthy_model.h5', compile=False)
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

    # Improved green range (handles more cases)
    lower = np.array([20, 30, 30])
    upper = np.array([95, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # Noise removal
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    leaves = []
    boxes = []

    for c in contours:
        area = cv2.contourArea(c)

        # Ignore small noise
        if area < 1000:
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

    img = np.array(image).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    p = model.predict(img, verbose=0)[0][0]

    return p

# ===== INPUT OPTION =====
option = st.radio("Choose Input Method:", ["Upload Image", "Use Camera"])

image = None

# ===== Upload Option =====
if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "📤 Upload Leaf Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(BytesIO(uploaded_file.read())).convert("RGB")

# ===== Camera Option =====
elif option == "Use Camera":
    camera_file = st.camera_input("📷 Take a photo")

    if camera_file is not None:
        image = Image.open(BytesIO(camera_file.read())).convert("RGB")

# ===== PROCESS =====
if image is not None:
    try:
        st.image(image, caption="Original Image", use_column_width=True)

        # Detect multiple leaves
        leaves, boxes = detect_leaves(image)

        if len(leaves) == 0:
            st.warning("⚠️ No leaf detected. Trying full image...")
            leaves = [image]

        result_img = np.array(image)

        st.subheader("🔍 Leaf-wise Prediction")

        for i, (leaf, box) in enumerate(zip(leaves, boxes if boxes else [(0,0,0,0)])):
            p = predict_image(leaf)

            if p >= 0.5:
                label = "Unhealthy"
                confidence = p * 100
                color = (0, 0, 255)
            else:
                label = "Healthy"
                confidence = (1 - p) * 100
                color = (0, 255, 0)

            st.image(leaf, caption=f"Leaf {i+1}")

            st.write(f"Prediction: {label}")
            st.write(f"Confidence: {confidence:.2f}%")
            st.write(f"Raw Value: {p:.4f}")

            # Draw bounding box if available
            if boxes:
                x, y, w, h = box
                cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(result_img, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        st.subheader("📦 Final Output")
        st.image(result_img, caption="Detected Leaves with Prediction", use_column_width=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")
def load_image(image_file):
    image = Image.open(image_file).convert("RGB")  # FIX: ensure RGB
    return image

def predict(image):
    image = np.array(image)  # FIX: convert PIL → NumPy

    image = cv2.resize(image, (224, 224))  # Resize
    image = image / 255.0  # FIX: normalization

    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)

    return predictions

# Streamlit UI
st.title('Plant Health Detection')
st.write('Upload an image of a plant leaf to check its health.')

# Image upload
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    predictions = predict(image)
    st.write('Prediction:', predictions)

# Webcam capture
if st.button('Capture from Webcam'):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # FIX: correct color
        st.image(frame_rgb, caption='Captured Image', use_container_width=True)

        predictions = predict(frame_rgb)
        st.write('Prediction:', predictions)

    cap.release()
