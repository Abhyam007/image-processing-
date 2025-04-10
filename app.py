import streamlit as st
import numpy as np
import cv2
from PIL import Image

# 💅 Styling
st.markdown("""
    <style>
    html, body, .stApp {
        height: 100%;
        margin: 0;
        background: linear-gradient(180deg, #f8b500 0%, #fceabb 33%, #ff9a9e 66%, #fad0c4 100%);
        font-family: 'Segoe UI', sans-serif;
        color: #333;
    }
    .main {
        background: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
    }
    .card-container {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 20px;
        margin-top: 2rem;
    }
    .card {
        background: #ffffff10;
        border: 2px solid #ffffff33;
        backdrop-filter: blur(10px);
        border-radius: 20px;
        width: 250px;
        height: 300px;
        perspective: 1000px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    }
    .card-inner {
        position: relative;
        width: 100%;
        height: 100%;
        transition: transform 0.8s;
        transform-style: preserve-3d;
    }
    .card:hover .card-inner {
        transform: rotateY(180deg);
    }
    .card-front, .card-back {
        position: absolute;
        width: 100%;
        height: 100%;
        backface-visibility: hidden;
        border-radius: 20px;
        padding: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        flex-direction: column;
    }
    .card-front {
        background: linear-gradient(to top left, #36D1DC, #5B86E5);
        color: white;
    }
    .card-front i {
        font-size: 40px;
        margin-bottom: 10px;
    }
    .card-back {
        background: linear-gradient(to bottom right, #ffe29f, #ffa99f);
        color: #333;
        transform: rotateY(180deg);
    }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

# 🧠 Processing Functions
def apply_smoothing(img, k=5): return cv2.GaussianBlur(img, (k, k), 0)
def apply_median_blur(img, k=5): return cv2.medianBlur(img, k)
def apply_bilateral_filter(img, k=9): return cv2.bilateralFilter(img, k, 75, 75)
def apply_sharpening(img):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)
def apply_contrast_stretch(img):
    in_min, in_max = np.percentile(img, 5), np.percentile(img, 95)
    stretched = (img - in_min) * (255 / (in_max - in_min))
    return np.clip(stretched, 0, 255).astype(np.uint8)
def apply_edge_detection(img, k=3):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
    edge = cv2.magnitude(sx, sy)
    return np.clip(edge, 0, 255).astype(np.uint8)
def apply_log(img):
    img = np.float32(img) + 1
    log = np.log(img) * (255 / np.log(256))
    return np.clip(log, 0, 255).astype(np.uint8)
def apply_hist_eq(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.equalizeHist(gray)
def apply_adaptive_thresh(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
def apply_emboss(img):
    kernel = np.array([[ -2, -1, 0],
                       [ -1,  1, 1],
                       [  0,  1, 2]])
    embossed = cv2.filter2D(img, -1, kernel) + 128
    return np.clip(embossed, 0, 255).astype(np.uint8)

# 🖼 Upload
st.title("🎨 Advanced Image Filters Playground")
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)
    st.image(img_np, caption="Original", use_column_width=True)

    st.subheader("✨ Choose a filter")
    filters = {
        "Smoothing": (apply_smoothing, True),
        "Median Blur": (apply_median_blur, True),
        "Bilateral Filter": (apply_bilateral_filter, True),
        "Sharpening": (apply_sharpening, False),
        "Contrast Stretching": (apply_contrast_stretch, False),
        "Edge Detection (Sobel)": (apply_edge_detection, True),
        "Log Transformation": (apply_log, False),
        "Histogram Equalization": (apply_hist_eq, False),
        "Adaptive Thresholding": (apply_adaptive_thresh, False),
        "Emboss": (apply_emboss, False)
    }

    col1, col2 = st.columns(2)

    with col1:
        filter_name = st.selectbox("🧪 Select a Filter", list(filters.keys()))
    with col2:
        k = 5
        if filters[filter_name][1]:
            k = st.slider("Kernel Size (odd)", 3, 15, 5, 2)

    # ✅ Apply
    if st.button("Apply Filter"):
        func = filters[filter_name][0]
        output = func(img_np, k) if filters[filter_name][1] else func(img_np)
        mode = "RGB" if len(output.shape) == 3 else "GRAY"
        st.image(output, caption=f"{filter_name} Result", use_column_width=True, channels=mode)
