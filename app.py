import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
import base64
import requests
import os
from models import UNetEnhancer

# Set Streamlit Page Configuration
st.set_page_config(page_title="Medical X-ray Enhancement & Fracture Detection", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------------
# 1. Model Loader
# -------------------------------------------------------------
@st.cache_resource
def load_models():
    # Load U-Net Enhancer
    enhancer = UNetEnhancer().to(device)
    if os.path.exists("unet_enhancer.pth"):
        enhancer.load_state_dict(torch.load("unet_enhancer.pth", map_location=device))
    enhancer.eval()

    # Load ResNet-18 Classifier
    classifier = resnet18()
    classifier.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    classifier.fc = nn.Linear(classifier.fc.in_features, 2)
    if os.path.exists("fracture_cnn.pth"):
        classifier.load_state_dict(torch.load("fracture_cnn.pth", map_location=device))
    classifier.eval()

    return enhancer, classifier

# Instantiate models globally so enhance_image and predict_fracture can access them
enhancer_model, classifier_model = load_models()

# -------------------------------------------------------------
# 2. Preprocessing & Postprocessing
# -------------------------------------------------------------
def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    
    # 1. Handle photometric inversion (white background -> black background)
    corners = [img_gray[0, 0], img_gray[0, -1], img_gray[-1, 0], img_gray[-1, -1]]
    if np.mean(corners) > 127:
        img_gray = cv2.bitwise_not(img_gray)

    # 2. Maintain higher resolution (keep original aspect ratio up to 512x512)
    # The U-Net dimensions must be divisible by 16 (for 4 downsampling layers)
    h, w = img_gray.shape
    scale = min(512 / h, 512 / w, 1.0)
    new_h = int((h * scale) // 16 * 16)
    new_w = int((w * scale) // 16 * 16)
    
    img_resized = cv2.resize(img_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img_norm = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)
    
    return img_resized, img_tensor

def enhance_image(tensor):
    with torch.no_grad():
        out = enhancer_model(tensor)
        out_np = out.squeeze().cpu().numpy()
        unet_enhanced = (out_np * 255.0).clip(0, 255).astype(np.uint8)

    orig_np = (tensor.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    # 1. Residual Detail Fusion: Combine U-Net denoised output with the original crisp details
    # This prevents the blurry / cartoonish look
    blended = cv2.addWeighted(orig_np, 0.45, unet_enhanced, 0.55, 0)

    # 2. Medical-grade CLAHE with smaller tiles for crisp trabecular pattern
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(6, 6))
    clahe_out = clahe.apply(blended)

    # 3. High-pass Laplacian edge sharpening (avoids gaussian halos)
    laplacian = cv2.Laplacian(clahe_out, cv2.CV_64F)
    sharp = clahe_out - 0.25 * laplacian
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)

    return sharp

def predict_fracture(enhanced_img):
    img_norm = enhanced_img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)
    # ImageNet normalization matching training
    tensor = (tensor - 0.485) / 0.229

    with torch.no_grad():
        logits = classifier_model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    prob_fractured = float(probs[0] * 100.0)
    prob_normal = float(probs[1] * 100.0)

    # Class 0: Fractured, Class 1: Normal
    classes = ["Fractured", "Normal"]
    idx = int(np.argmax(probs))
    return classes[idx], probs[idx] * 100.0, prob_fractured, prob_normal

def query_llava(image_np, prediction, confidence):
    _, buffer = cv2.imencode('.png', image_np)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    prompt = (
        f"The CAD system classified this X-ray as '{prediction}' "
        f"with {confidence:.2f}% confidence. Provide a structured medical "
        f"interpretation describing the visible bone cortical margins, alignment, "
        f"and precautionary medical recommendations. Conclude with an educational disclaimer."
    )

    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llava",
                "prompt": prompt,
                "images": [img_base64],
                "stream": False
            },
            timeout=45
        )
        if res.status_code == 200:
            return res.json().get("response", "No response received.")
    except Exception:
        pass

    return (
        f"This X-ray analysis indicates a state of '{prediction}' ({confidence:.2f}% certainty). "
        f"The visual analysis highlights bone continuity and joint spacing. In case of disruption "
        f"or cortical line discontinuity, bone fracture or displacement may be present. "
        f"Keep the area immobilized and seek review from an orthopedic radiologist."
    )

# -------------------------------------------------------------
# 3. Streamlit Interface
# -------------------------------------------------------------
st.title("Medical X-ray Image Enhancement System")

uploaded_file = st.file_uploader(
    "Upload a low-quality X-ray image to view the enhanced result", 
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    if "original" not in st.session_state or st.session_state.get("filename") != uploaded_file.name:
        orig_img, tensor = preprocess_image(uploaded_file)
        enhanced_img = enhance_image(tensor)
        st.session_state.original = orig_img
        st.session_state.enhanced = enhanced_img
        st.session_state.filename = uploaded_file.name
        st.session_state.prediction = None
        st.session_state.confidence = None
        st.session_state.prob_frac = None
        st.session_state.prob_norm = None
        st.session_state.explanation = None

    # Side-by-side comparison
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(st.session_state.original, clamp=True, use_container_width=True)
    with col2:
        st.subheader("Enhanced Image")
        st.image(st.session_state.enhanced, clamp=True, use_container_width=True)

    st.markdown("---")

    # Classification Button
    if st.button("Detect Fracture"):
        label, conf, p_frac, p_norm = predict_fracture(st.session_state.enhanced)
        st.session_state.prediction = label
        st.session_state.confidence = conf
        st.session_state.prob_frac = p_frac
        st.session_state.prob_norm = p_norm

    if st.session_state.prediction is not None:
        st.subheader("Prediction Result")
        
        if st.session_state.prediction == "Fractured":
            st.error(f"**Prediction:** {st.session_state.prediction}")
        else:
            st.success(f"**Prediction:** {st.session_state.prediction}")
            
        st.write(f"**Confidence:** {st.session_state.confidence:.2f}%")
        st.caption(f"Score Breakdown — Fractured: {st.session_state.prob_frac:.2f}% | Normal: {st.session_state.prob_norm:.2f}%")

        if st.button("Generate AI Explanation"):
            with st.spinner("Generating LLM-based clinical interpretation..."):
                explanation = query_llava(
                    st.session_state.enhanced, 
                    st.session_state.prediction, 
                    st.session_state.confidence
                )
                st.session_state.explanation = explanation

    if st.session_state.explanation is not None:
        st.subheader("AI Medical Explanation")
        st.write(st.session_state.explanation)