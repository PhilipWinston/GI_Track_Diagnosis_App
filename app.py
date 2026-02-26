import os
import time
import tempfile
from typing import List, Tuple
import streamlit as st
import torch
import timm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from torchvision import transforms
import torch.nn as nn
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import gdown

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except:
    HAS_ULTRALYTICS = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="GI Tract Diagnosis System",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Deep Learning Framework for GI Tract Disorder Diagnosis"
    }
)

# ============================================================================
# CONSTANTS
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
DISPLAY_WIDTH = 500

CLASS_NAMES = ["Normal", "Ulcerative Colitis", "Polyps", "Esophagitis"]
CLASS_COLORS = {
    0: "#10b981",  # Green - Normal
    1: "#f59e0b",  # Amber - UC
    2: "#3b82f6",  # Blue - Polyps
    3: "#8b5cf6"   # Purple - Esophagitis
}
CLASS_ICONS = {
    0: "✓",
    1: "⚠",
    2: "●",
    3: "!"
}

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

DRIVE_IDS = {
    "classifier": "1cvDPCfVBLWjCtx9mjz0KFWCwTwRs2atL",
    "ordinal": "1Hvng1F6upAUjfsZe_sLpTmmH8lia04TI",
    "polyp": "1xzGUJ1d9qDQKiodCzbWVOH07gNsffWnS",
}

MODEL_PATHS = {
    k: os.path.join(MODEL_DIR, f"best_{k}.pth" if k != "polyp" else "best.pt")
    for k in DRIVE_IDS
}

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    /* Remove default padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2.5rem 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: white !important;
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        color: white !important;
        opacity: 0.95;
    }
    
    /* Card styling */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        margin-bottom: 1.5rem;
        border: 1px solid #e5e7eb;
        height: 100%;
    }
    
    .image-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border: 1px solid #e5e7eb;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .image-card h3 {
        margin: 0 0 1rem 0;
        color: #1f2937 !important;
        font-size: 1.3rem;
    }
    
    /* Prediction badge */
    .prediction-badge {
        display: inline-block;
        padding: 1rem 2rem;
        border-radius: 30px;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        color: white !important;
    }
    
    /* Confidence container */
    .confidence-container {
        margin: 2rem 0;
    }
    
    .confidence-label {
        font-size: 1rem;
        color: #374151 !important;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    
    /* Severity badge */
    .severity-badge {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 1rem 0;
        color: white !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Detection info */
    .detection-info {
        background: #dbeafe;
        border-left: 5px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 10px;
        margin: 1.5rem 0;
    }
    
    .detection-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e3a8a !important;
        margin-bottom: 0.5rem;
    }
    
    .detection-subtitle {
        font-size: 0.95rem;
        color: #1e40af !important;
    }
    
    .detection-info-warning {
        background: #fef3c7;
        border-left: 5px solid #f59e0b;
    }
    
    .detection-info-warning .detection-title {
        color: #78350f !important;
    }
    
    .detection-info-warning .detection-subtitle {
        color: #92400e !important;
    }
    
    /* Info boxes */
    .info-box {
        background: #f9fafb;
        border-radius: 10px;
        padding: 1.25rem;
        margin: 1.5rem 0;
        border: 1px solid #e5e7eb;
    }
    
    .info-box-title {
        font-weight: 700;
        color: #1f2937 !important;
        margin-bottom: 0.75rem;
        font-size: 1.1rem;
    }
    
    /* Stats display */
    .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 0.75rem 0;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .stat-label {
        color: #6b7280 !important;
        font-size: 1rem;
    }
    
    .stat-value {
        color: #111827 !important;
        font-weight: 600;
        font-size: 1rem;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(to right, transparent, #e5e7eb, transparent);
        margin: 3rem 0;
    }
    
    /* Upload section */
    .upload-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .upload-section h3 {
        color: #1f2937 !important;
        margin-top: 0;
    }
    
    .upload-section p {
        color: #4b5563 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #6b7280 !important;
        font-size: 0.95rem;
        margin-top: 3rem;
        border-top: 1px solid #e5e7eb;
    }
    
    /* Streamlit overrides */
    [data-testid="column"] {
        padding: 0.5rem;
    }
    
    /* Make sure all text is visible */
    p, span, div, h1, h2, h3, h4, h5, h6 {
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL CLASSES
# ============================================================================
class EFFResNetViT(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.eff = timm.create_model("efficientnet_b4", pretrained=False, features_only=True)
        self.res = timm.create_model("resnet50", pretrained=False, features_only=True)
        
        eff_dim = self.eff.feature_info[-1]["num_chs"]
        res_dim = self.res.feature_info[-1]["num_chs"]
        
        self.fusion = nn.Conv2d(eff_dim + res_dim, 768, kernel_size=1)
        
        enc = nn.TransformerEncoderLayer(
            d_model=768, nhead=12, dim_feedforward=3072,
            dropout=0.2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=3)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.4),
            nn.Linear(768, num_classes)
        )
    
    def forward(self, x):
        eff = self.eff(x)[-1]
        res = self.res(x)[-1]
        fused = torch.cat([eff, res], dim=1)
        t = self.fusion(fused)
        t = t.flatten(2).transpose(1, 2)
        t = self.transformer(t)
        pooled = t.mean(dim=1)
        return self.classifier(pooled)


class EFFResNetViTOrdinal(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.eff = timm.create_model("efficientnet_b4", pretrained=False, features_only=True)
        self.res = timm.create_model("resnet50", pretrained=False, features_only=True)
        
        eff_dim = self.eff.feature_info[-1]["num_chs"]
        res_dim = self.res.feature_info[-1]["num_chs"]
        
        self.fusion = nn.Conv2d(eff_dim + res_dim, 768, kernel_size=1)
        
        enc = nn.TransformerEncoderLayer(
            d_model=768, nhead=12, dim_feedforward=3072,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=2)
        self.ordinal_head = nn.Linear(768, num_classes - 1)
    
    def forward(self, x):
        eff = self.eff(x)[-1]
        res = self.res(x)[-1]
        fused = torch.cat([eff, res], dim=1)
        t = self.fusion(fused)
        t = t.flatten(2).transpose(1, 2)
        t = self.transformer(t)
        pooled = t.mean(dim=1)
        return self.ordinal_head(pooled)


# ============================================================================
# TRANSFORM
# ============================================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def download_if_missing(file_id: str, out_path: str, desc: str):
    """Download model from Google Drive if not present"""
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        return True
    
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        with st.spinner(f"Downloading {desc} model..."):
            gdown.download(url, out_path, quiet=False)
        return os.path.exists(out_path) and os.path.getsize(out_path) > 1024
    except:
        return False


@st.cache_resource
def load_classification_model(path: str):
    """Load classification model"""
    m = EFFResNetViT(num_classes=4).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


@st.cache_resource
def load_ordinal_model(path: str):
    """Load ordinal severity model"""
    m = EFFResNetViTOrdinal(num_classes=4).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


@st.cache_resource
def load_yolo_model(path: str):
    """Load YOLO polyp detection model"""
    if not HAS_ULTRALYTICS:
        return (None, None)
    try:
        return ("ultralytics", YOLO(path))
    except:
        return (None, None)


def predict_class(model, pil_img):
    """Predict class from image"""
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(probs.argmax())
        conf = float(probs[pred])
    return pred, conf, probs.cpu().numpy()


def predict_ordinal(ordinal_model, pil_img):
    """Predict ordinal severity using cumulative link model"""
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = ordinal_model(x)
        # Cumulative probabilities for ordinal regression
        # These represent P(Y > threshold_k)
        cumulative_probs = torch.sigmoid(logits)[0].cpu().numpy()
        
        # Convert cumulative probabilities to individual class probabilities
        # For ordinal regression with 4 classes (0,1,2,3), we have 3 thresholds
        # P(Y=0) = 1 - P(Y>0) = 1 - cumulative_probs[0]
        # P(Y=1) = P(Y>0) - P(Y>1) = cumulative_probs[0] - cumulative_probs[1]
        # P(Y=2) = P(Y>1) - P(Y>2) = cumulative_probs[1] - cumulative_probs[2]
        # P(Y=3) = P(Y>2) = cumulative_probs[2]
        
        individual_probs = np.zeros(4)
        individual_probs[0] = 1.0 - cumulative_probs[0]  # Remission (class 0)
        individual_probs[1] = cumulative_probs[0] - cumulative_probs[1]  # Mild (class 1)
        individual_probs[2] = cumulative_probs[1] - cumulative_probs[2]  # Moderate (class 2)
        individual_probs[3] = cumulative_probs[2]  # Severe (class 3)
        
        # Ensure probabilities are non-negative and sum to 1
        individual_probs = np.clip(individual_probs, 0, 1)
        individual_probs = individual_probs / individual_probs.sum()
        
        # Predicted class is the one with highest probability
        label = int(np.argmax(individual_probs))
        
    return label, individual_probs


def run_yolo(yolo_loader, src_image, conf=0.25):
    """Run YOLO polyp detection"""
    if yolo_loader[0] != "ultralytics":
        return []
    
    # Use higher resolution for detection
    img_bgr = cv2.cvtColor(
        np.array(src_image.resize((640, 640))), 
        cv2.COLOR_RGB2BGR
    )
    
    results = yolo_loader[1].predict(source=img_bgr, conf=conf, verbose=False)
    r = results[0]
    
    boxes = []
    for b in r.boxes.data.tolist():
        x1, y1, x2, y2, confv, cls = b
        boxes.append({
            "xyxy": np.array([x1, y1, x2, y2]),
            "conf": confv
        })
    
    return boxes


def overlay_boxes_high_quality(image, boxes):
    """Overlay detection boxes with high quality rendering"""
    # Convert to RGB
    img_pil = image.convert("RGB")
    w, h = img_pil.size
    
    # Create drawing object
    draw = ImageDraw.Draw(img_pil)
    
    # Calculate font size (smaller and responsive)
    font_size = max(14, min(w, h) // 35)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
    
    # Scale factors from YOLO resolution (640x640) to image size
    sx = w / 640
    sy = h / 640
    
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box["xyxy"]
        
        # Scale to image coordinates
        x1 = max(0, int(x1 * sx))
        y1 = max(0, int(y1 * sy))
        x2 = min(w, int(x2 * sx))
        y2 = min(h, int(y2 * sy))
        
        # Draw bounding box with green color
        line_width = max(3, min(w, h) // 200)
        for i in range(line_width):
            draw.rectangle(
                [x1 + i, y1 + i, x2 - i, y2 - i],
                outline=(0, 255, 0),
                width=1
            )
        
        # Create label text - shorter format
        label = f"Polyp {idx + 1} - {box['conf']*100:.0f}%"
        
        # Get text dimensions
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # Add padding
        pad = 6
        label_w = text_w + 2 * pad
        label_h = text_h + 2 * pad
        
        # Position label (above box if space, otherwise below)
        if y1 - label_h - 5 > 0:
            # Above the box
            label_x = x1
            label_y = y1 - label_h - 5
        elif y2 + label_h + 5 < h:
            # Below the box
            label_x = x1
            label_y = y2 + 5
        else:
            # Inside the box at top
            label_x = x1 + 5
            label_y = y1 + 5
        
        # Make sure label doesn't go off screen
        if label_x + label_w > w:
            label_x = w - label_w - 5
        label_x = max(5, label_x)
        
        # Draw label background (green)
        draw.rectangle(
            [label_x, label_y, label_x + label_w, label_y + label_h],
            fill=(0, 200, 0),
            outline=(0, 150, 0),
            width=2
        )
        
        # Draw label text (white)
        draw.text(
            (label_x + pad, label_y + pad),
            label,
            fill=(255, 255, 255),
            font=font
        )
    
    return img_pil


# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Header
st.markdown("""
<div class="main-header">
    <h1>🔬 GI Tract Diagnosis System</h1>
    <p>Deep Learning Framework for Gastrointestinal Disorder Detection</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Configuration")

conf_threshold = st.sidebar.slider(
    "YOLO Detection Confidence",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
    help="Minimum confidence threshold for polyp detection"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Status")

# Download and load models
dl_ok = {}
for k in DRIVE_IDS:
    ok = download_if_missing(DRIVE_IDS[k], MODEL_PATHS[k], k)
    dl_ok[k] = ok
    if ok:
        st.sidebar.success(f"✅ {k.capitalize()} Model")
    else:
        st.sidebar.error(f"❌ {k.capitalize()} Model")

# Load models
classification_model = None
ordinal_model = None
yolo_loader = (None, None)

if dl_ok.get("classifier") and os.path.exists(MODEL_PATHS["classifier"]):
    classification_model = load_classification_model(MODEL_PATHS["classifier"])

if dl_ok.get("ordinal") and os.path.exists(MODEL_PATHS["ordinal"]):
    ordinal_model = load_ordinal_model(MODEL_PATHS["ordinal"])

if dl_ok.get("polyp") and os.path.exists(MODEL_PATHS["polyp"]) and HAS_ULTRALYTICS:
    yolo_loader = load_yolo_model(MODEL_PATHS["polyp"])

# System info
st.sidebar.markdown("---")
st.sidebar.markdown("### 💻 System Info")
st.sidebar.info(f"**Device:** {DEVICE.type.upper()}")
st.sidebar.info(f"**Image Size:** {IMG_SIZE}×{IMG_SIZE}")

# Main content
st.markdown("""
<div class="info-box">
    <div class="info-box-title">📋 Supported Conditions</div>
    <div style="margin-top: 0.5rem;">
        <span style="color: #10b981; font-weight: 600;">● Normal</span> • 
        <span style="color: #f59e0b; font-weight: 600;">● Ulcerative Colitis</span> • 
        <span style="color: #3b82f6; font-weight: 600;">● Polyps</span> • 
        <span style="color: #8b5cf6; font-weight: 600;">● Esophagitis</span>
    </div>
</div>
""", unsafe_allow_html=True)

# File uploader
uploaded = st.file_uploader(
    "📤 Upload Colonoscopy Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Upload one or more colonoscopy images for analysis"
)

if uploaded:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("## 🔍 Analysis Results")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, f in enumerate(uploaded):
        status_text.text(f"Processing image {idx+1} of {len(uploaded)}...")
        
        img = Image.open(f).convert("RGB")
        
        # Create two columns
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.markdown(f"""
            <div class="image-card">
                <h3>📷 Image {idx+1}</h3>
            """, unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            if not classification_model:
                st.error("❌ Classification model not loaded")
                st.markdown('</div>', unsafe_allow_html=True)
                continue
            
            # Predict
            pred, conf, all_probs = predict_class(classification_model, img)
            color = CLASS_COLORS[pred]
            name = CLASS_NAMES[pred]
            icon = CLASS_ICONS[pred]
            
            # Display prediction
            st.markdown(f"""
            <div style="text-align: center;">
                <div class="prediction-badge" style="background-color: {color}; color: white;">
                    {icon} {name}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence
            st.markdown('<div class="confidence-container">', unsafe_allow_html=True)
            st.markdown('<div class="confidence-label">Confidence Score</div>', unsafe_allow_html=True)
            st.progress(conf, text=f"{conf*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # All class probabilities
            with st.expander("📊 View All Class Probabilities"):
                for i, prob in enumerate(all_probs):
                    st.write(f"**{CLASS_NAMES[i]}:** {prob*100:.2f}%")
                    st.progress(float(prob))
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional analyses based on prediction
        if pred == 2 and yolo_loader[0] == "ultralytics":
            # Polyp detection
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("### 🎯 Polyp Detection Results")
            
            boxes = run_yolo(yolo_loader, img, conf_threshold)
            
            if boxes:
                detection_col1, detection_col2 = st.columns([1, 1])
                
                with detection_col1:
                    st.markdown(f"""
                    <div class="detection-info">
                        <div class="detection-title">✓ {len(boxes)} Polyp(s) Detected</div>
                        <div class="detection-subtitle">Confidence threshold: {conf_threshold*100:.0f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show detection stats
                    for i, box in enumerate(boxes):
                        st.metric(
                            f"Polyp #{i+1}",
                            f"{box['conf']*100:.1f}%",
                            delta="Detected"
                        )
                
                with detection_col2:
                    over = overlay_boxes_high_quality(img, boxes)
                    st.image(over, caption="Detected Polyps", use_container_width=True)
                    
                    # Download button
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    over.save(tmp.name, quality=95)
                    with open(tmp.name, "rb") as file:
                        st.download_button(
                            label="📥 Download Detection Image",
                            data=file,
                            file_name=f"polyp_detection_{idx+1}.png",
                            mime="image/png",
                            key=f"download_detection_{idx}"
                        )
            else:
                st.markdown("""
                <div class="detection-info detection-info-warning">
                    <div class="detection-title">ℹ️ No polyps detected</div>
                    <div class="detection-subtitle">Try adjusting the confidence threshold in the sidebar</div>
                </div>
                """, unsafe_allow_html=True)
        
        elif pred == 2:
            st.info("⚠️ YOLO model unavailable for polyp detection")
        
        # UC severity assessment
        if pred == 1 and ordinal_model:
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("### 📈 Ulcerative Colitis Severity Assessment")
            
            label, probs = predict_ordinal(ordinal_model, img)
            severity_names = ["Remission", "Mild", "Moderate", "Severe"]
            severity_colors = ["#10b981", "#f59e0b", "#fb923c", "#ef4444"]
            
            severity_col1, severity_col2 = st.columns([1, 1])
            
            with severity_col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 1.5rem;">
                    <div class="severity-badge" style="background-color: {severity_colors[label]};">
                        {severity_names[label]}
                    </div>
                    <div style="margin-top: 1.5rem; font-size: 1.2rem; color: #1f2937; font-weight: 600;">
                        Severity Grade: <strong style="color: {severity_colors[label]};">{label}/3</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with severity_col2:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown('<div class="info-box-title">Severity Class Probabilities</div>', unsafe_allow_html=True)
                
                for i, (p, name) in enumerate(zip(probs, severity_names)):
                    # Add checkmark to predicted class
                    prefix = "✓ " if i == label else "   "
                    st.markdown(f"""
                    <div style="margin: 0.75rem 0;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                            <span style="font-weight: {'700' if i == label else '600'}; color: {'#1f2937' if i == label else '#6b7280'};">
                                {prefix}{name} (Grade {i})
                            </span>
                            <span style="font-weight: 700; color: {severity_colors[i]};">{p*100:.2f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(p))
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Separator between images
        if idx < len(uploaded) - 1:
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        progress_bar.progress((idx + 1) / len(uploaded))
    
    status_text.text("✓ Analysis complete!")
    progress_bar.empty()
    status_text.empty()

else:
    st.markdown("""
    <div class="upload-section">
        <h3 style="color: #374151; margin-top: 0;">👆 Upload Images to Begin</h3>
        <p style="color: #6b7280; margin-bottom: 0;">
            Upload colonoscopy images in JPG or PNG format for automated diagnosis
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <strong>Deep Learning Framework for GI Tract Disorder Diagnosis</strong><br>
    Final Year Project 2026 • EFFResNetViT Architecture with YOLO Detection
</div>
""", unsafe_allow_html=True)
