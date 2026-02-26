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
    /* Main styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: white !important;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.95;
        margin: 0;
    }
    
    /* Card styling */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        margin-bottom: 1.5rem;
        border: 1px solid #e5e7eb;
    }
    
    .image-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    
    /* Prediction badge */
    .prediction-badge {
        display: inline-block;
        padding: 0.75rem 1.5rem;
        border-radius: 25px;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Confidence bar */
    .confidence-container {
        margin: 1.5rem 0;
    }
    
    .confidence-label {
        font-size: 0.9rem;
        color: #6b7280;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }
    
    /* Severity badge */
    .severity-badge {
        display: inline-block;
        padding: 0.6rem 1.2rem;
        border-radius: 20px;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0.5rem 0;
        color: white;
    }
    
    /* Detection info */
    .detection-info {
        background: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .detection-info-warning {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
    }
    
    /* Info boxes */
    .info-box {
        background: #f9fafb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #e5e7eb;
    }
    
    .info-box-title {
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
    }
    
    /* Stats display */
    .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .stat-label {
        color: #6b7280;
        font-size: 0.9rem;
    }
    
    .stat-value {
        color: #111827;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Button styling */
    .stDownloadButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f9fafb;
    }
    
    /* Upload section */
    .upload-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(to right, transparent, #e5e7eb, transparent);
        margin: 2rem 0;
    }
    
    /* Image container */
    .image-container {
        position: relative;
        display: inline-block;
        margin: 1rem 0;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #6b7280;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid #e5e7eb;
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
    """Predict ordinal severity"""
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = ordinal_model(x)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        label = int((probs > 0.5).sum())
    return label, probs


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
    # Work with original image resolution
    img_pil = image.convert("RGB")
    draw = ImageDraw.Draw(img_pil)
    
    # Try to load a nice font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Scale factors from detection resolution (640x640) to original
    w, h = img_pil.size
    sx = w / 640
    sy = h / 640
    
    for idx, b in enumerate(boxes):
        x1, y1, x2, y2 = b["xyxy"]
        
        # Scale coordinates to original image size
        x1 = int(x1 * sx)
        y1 = int(y1 * sy)
        x2 = int(x2 * sx)
        y2 = int(y2 * sy)
        
        # Draw semi-transparent fill
        overlay = Image.new('RGBA', img_pil.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([x1, y1, x2, y2], fill=(59, 130, 246, 40))
        img_pil = Image.alpha_composite(img_pil.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img_pil)
        
        # Draw thick border
        line_width = max(3, int(min(w, h) * 0.005))
        for offset in range(line_width):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=(59, 130, 246),
                width=1
            )
        
        # Draw label background
        label_text = f"Polyp #{idx + 1} ({b['conf']*100:.1f}%)"
        
        # Get text size
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        label_y = y1 - text_height - 10 if y1 - text_height - 10 > 0 else y2 + 5
        
        # Draw label background with padding
        padding = 8
        draw.rectangle(
            [x1, label_y - padding, x1 + text_width + 2*padding, label_y + text_height + padding],
            fill=(59, 130, 246),
            outline=(37, 99, 235),
            width=2
        )
        
        # Draw text
        draw.text(
            (x1 + padding, label_y),
            label_text,
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
                <h3 style="margin-top: 0; color: #374151;">📷 Image {idx+1}</h3>
            </div>
            """, unsafe_allow_html=True)
            st.image(img, use_container_width=True)
        
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
                        <strong>✓ {len(boxes)} Polyp(s) Detected</strong><br>
                        <small>Confidence threshold: {conf_threshold*100:.0f}%</small>
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
                    <strong>ℹ️ No polyps detected above confidence threshold</strong><br>
                    <small>Try adjusting the confidence threshold in the sidebar</small>
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
                <div style="text-align: center; padding: 1rem;">
                    <div class="severity-badge" style="background-color: {severity_colors[label]};">
                        {severity_names[label]}
                    </div>
                    <div style="margin-top: 1rem; font-size: 1.1rem; color: #6b7280;">
                        Severity Grade: <strong>{label}/3</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with severity_col2:
                st.markdown("**Ordinal Probabilities:**")
                for i, p in enumerate(probs):
                    st.write(f"Stage {i+1}: {p*100:.2f}%")
                    st.progress(float(p))
        
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
