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
    "classifier": "1aZke_47izApUtev2-Jlr1j4ZC84DC1i1",
    "ordinal": "1Q74a7he0LnLfDJEN90YLpMhzJxI0wKa2",
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
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
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
    .confidence-container {
        margin: 2rem 0;
    }
    .confidence-label {
        font-size: 1rem;
        color: #374151 !important;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
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
    .divider {
        height: 2px;
        background: linear-gradient(to right, transparent, #e5e7eb, transparent);
        margin: 3rem 0;
    }
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
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #6b7280 !important;
        font-size: 0.95rem;
        margin-top: 3rem;
        border-top: 1px solid #e5e7eb;
    }
    [data-testid="column"] {
        padding: 0.5rem;
    }
    p, span, div, h1, h2, h3, h4, h5, h6 {
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL CLASSES
# ============================================================================

class ResEffFusionClassifier(nn.Module):
    """4-class classifier: Normal, UC, Polyps, Esophagitis"""
    def __init__(self, num_classes=4, eff_weight=0.75):
        super().__init__()
        self.eff_weight = eff_weight
        self.res_weight = 1 - eff_weight

        self.eff = timm.create_model("efficientnet_b4", pretrained=False, features_only=True)
        eff_dim = self.eff.feature_info[-1]['num_chs']

        self.res = timm.create_model("resnet50", pretrained=False, features_only=True)
        res_dim = self.res.feature_info[-1]['num_chs']

        self.eff_proj = nn.Conv2d(eff_dim, 1024, 1)
        self.res_proj = nn.Conv2d(res_dim, 1024, 1)

        self.bn   = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        eff = self.eff(x)[-1]
        res = self.res(x)[-1]

        if eff.shape[2:] != res.shape[2:]:
            res = nn.functional.interpolate(res, size=eff.shape[2:], mode="bilinear", align_corners=False)

        eff = self.eff_proj(eff)
        res = self.res_proj(res)

        fused = self.eff_weight * eff + self.res_weight * res
        fused = self.relu(self.bn(fused))

        pooled = self.pool(fused).flatten(1)
        return self.classifier(pooled)


class ResEffFusionOrdinal(nn.Module):
    """Ordinal (CORAL) severity model: outputs K-1 logits for 4 grades"""
    def __init__(self, num_classes=4, eff_weight=0.75):
        super().__init__()
        self.num_classes = num_classes
        self.eff_weight  = eff_weight
        self.res_weight  = 1 - eff_weight

        self.eff = timm.create_model("efficientnet_b4", pretrained=False, features_only=True)
        eff_dim = self.eff.feature_info[-1]['num_chs']

        self.res = timm.create_model("resnet50", pretrained=False, features_only=True)
        res_dim = self.res.feature_info[-1]['num_chs']

        self.eff_proj = nn.Conv2d(eff_dim, 1024, 1)
        self.res_proj = nn.Conv2d(res_dim, 1024, 1)

        self.bn   = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # CORAL head: K-1 outputs
        self.classifier = nn.Linear(1024, num_classes - 1)

    def forward(self, x):
        eff = self.eff(x)[-1]
        res = self.res(x)[-1]

        if eff.shape[2:] != res.shape[2:]:
            res = nn.functional.interpolate(res, size=eff.shape[2:], mode="bilinear", align_corners=False)

        eff = self.eff_proj(eff)
        res = self.res_proj(res)

        fused = self.eff_weight * eff + self.res_weight * res
        fused = self.relu(self.bn(fused))

        pooled = self.pool(fused).flatten(1)
        return self.classifier(pooled)


# ============================================================================
# CORAL UTILITY
# ============================================================================

def coral_logits_to_probs(logits: torch.Tensor) -> np.ndarray:
    """Convert CORAL K-1 logits to K class probabilities."""
    sig  = torch.sigmoid(logits)          # (1, K-1)
    K_1  = logits.size(1)
    K    = K_1 + 1

    probs = []
    probs.append((1 - sig[:, 0]).unsqueeze(1))          # P(Y=0)
    for r in range(1, K - 1):
        probs.append((sig[:, r-1] - sig[:, r]).unsqueeze(1))
    probs.append(sig[:, K_1 - 1].unsqueeze(1))           # P(Y=K-1)

    probs = torch.cat(probs, dim=1)
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-12)
    return probs.squeeze(0).cpu().numpy()


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
    m = ResEffFusionClassifier(num_classes=4).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


@st.cache_resource
def load_ordinal_model(path: str):
    m = ResEffFusionOrdinal(num_classes=4).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m


@st.cache_resource
def load_yolo_model(path: str):
    if not HAS_ULTRALYTICS:
        return (None, None)
    try:
        return ("ultralytics", YOLO(path))
    except:
        return (None, None)


def predict_class(model, pil_img):
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = int(probs.argmax())
        conf   = float(probs[pred])
    return pred, conf, probs.cpu().numpy()


def predict_ordinal(ordinal_model, pil_img):
    """CORAL ordinal prediction → label + per-class probabilities."""
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = ordinal_model(x)
        probs  = coral_logits_to_probs(logits)          # shape (4,)
        probs  = np.clip(probs, 0, 1)
        probs  = probs / probs.sum()
        label  = int(np.argmax(probs))
    return label, probs


def run_yolo(yolo_loader, src_image, conf=0.25):
    if yolo_loader[0] != "ultralytics":
        return []
    img_bgr = cv2.cvtColor(
        np.array(src_image.resize((640, 640))),
        cv2.COLOR_RGB2BGR
    )
    results = yolo_loader[1].predict(source=img_bgr, conf=conf, verbose=False)
    r = results[0]
    boxes = []
    for b in r.boxes.data.tolist():
        x1, y1, x2, y2, confv, cls = b
        boxes.append({"xyxy": np.array([x1, y1, x2, y2]), "conf": confv})
    return boxes


# ============================================================================
# IMPROVED YOLO BOUNDING BOX OVERLAY
# ============================================================================

def overlay_boxes_high_quality(image: Image.Image, boxes: list) -> Image.Image:
    """
    Render YOLO detections with a polished, clinical look:
    - Thick neon-green box with rounded corners drawn via multiple rects
    - Corner accent marks (L-shaped brackets)
    - Semi-transparent filled label pill
    - Confidence bar inside the label
    """
    img_pil = image.convert("RGBA")
    w, h    = img_pil.size

    # ---- overlay layer for transparent fills ----
    overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
    draw_o  = ImageDraw.Draw(overlay)
    draw    = ImageDraw.Draw(img_pil)

    # Responsive sizes
    lw        = max(3, min(w, h) // 180)       # box line width
    corner_sz = max(18, min(w, h) // 18)       # L-bracket arm length
    font_size = max(15, min(w, h) // 28)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", max(12, font_size - 4))
    except:
        try:
            font    = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            font_sm = font
        except:
            font    = ImageFont.load_default()
            font_sm = font

    # Scale factors from YOLO 640×640 → image size
    sx = w / 640
    sy = h / 640

    # Colour palette
    BOX_COLOR    = (0, 230, 100, 255)      # bright green  RGBA
    BRACKET_COL  = (255, 255, 255, 255)    # white corners
    LABEL_BG     = (0, 180, 80, 210)       # semi-transparent green pill
    LABEL_FG     = (255, 255, 255, 255)    # white text
    BAR_BG       = (0, 100, 40, 200)
    BAR_FG       = (160, 255, 160, 230)

    for idx, box in enumerate(boxes):
        x1r, y1r, x2r, y2r = box["xyxy"]
        x1 = max(0, int(x1r * sx))
        y1 = max(0, int(y1r * sy))
        x2 = min(w - 1, int(x2r * sx))
        y2 = min(h - 1, int(y2r * sy))

        bw = x2 - x1
        bh = y2 - y1
        if bw < 4 or bh < 4:
            continue

        conf_val = box["conf"]

        # ---- 1. Subtle filled background for the box interior ----
        draw_o.rectangle([x1, y1, x2, y2], fill=(0, 230, 100, 22))

        # ---- 2. Main bounding box (layered for thickness) ----
        for i in range(lw):
            draw.rectangle(
                [x1 + i, y1 + i, x2 - i, y2 - i],
                outline=BOX_COLOR[:3],
                width=1
            )

        # ---- 3. Corner L-brackets (white accent) ----
        cs = min(corner_sz, bw // 3, bh // 3)
        ct = max(2, lw + 1)
        corners = [
            # top-left
            [(x1, y1, x1 + cs, y1 + ct), (x1, y1, x1 + ct, y1 + cs)],
            # top-right
            [(x2 - cs, y1, x2, y1 + ct), (x2 - ct, y1, x2, y1 + cs)],
            # bottom-left
            [(x1, y2 - ct, x1 + cs, y2), (x1, y2 - cs, x1 + ct, y2)],
            # bottom-right
            [(x2 - cs, y2 - ct, x2, y2), (x2 - ct, y2 - cs, x2, y2)],
        ]
        for pair in corners:
            for rect in pair:
                draw.rectangle(rect, fill=BRACKET_COL[:3])

        # ---- 4. Label pill ----
        label_text = f"  Polyp {idx + 1}   {conf_val * 100:.0f}%  "
        bbox_t = draw.textbbox((0, 0), label_text, font=font)
        tw = bbox_t[2] - bbox_t[0]
        th = bbox_t[3] - bbox_t[1]

        pad_x, pad_y = 10, 6
        pill_w = tw + 2 * pad_x
        pill_h = th + 2 * pad_y + (font_size // 2)   # extra room for conf bar

        # Position: prefer above box
        if y1 - pill_h - 6 >= 0:
            px, py = x1, y1 - pill_h - 6
        elif y2 + pill_h + 6 <= h:
            px, py = x1, y2 + 6
        else:
            px, py = x1 + 5, y1 + 5

        # Clamp horizontally
        px = min(max(px, 0), w - pill_w - 2)

        # Draw pill background
        draw_o.rectangle([px, py, px + pill_w, py + pill_h], fill=LABEL_BG)

        # Draw text
        draw.text((px + pad_x, py + pad_y), label_text, fill=LABEL_FG[:3], font=font)

        # Confidence bar
        bar_y    = py + pad_y + th + 4
        bar_h    = max(5, font_size // 4)
        bar_x1   = px + pad_x
        bar_x2   = px + pill_w - pad_x
        fill_x2  = bar_x1 + int((bar_x2 - bar_x1) * conf_val)

        draw_o.rectangle([bar_x1, bar_y, bar_x2, bar_y + bar_h], fill=BAR_BG)
        if fill_x2 > bar_x1:
            draw_o.rectangle([bar_x1, bar_y, fill_x2, bar_y + bar_h], fill=BAR_FG)

    # Merge overlay
    img_pil = Image.alpha_composite(img_pil, overlay)
    return img_pil.convert("RGB")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

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
ordinal_model        = None
yolo_loader          = (None, None)

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

# Conditions info banner
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
    status_text  = st.empty()

    for idx, f in enumerate(uploaded):
        status_text.text(f"Processing image {idx+1} of {len(uploaded)}...")

        img = Image.open(f).convert("RGB")

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
            name  = CLASS_NAMES[pred]
            icon  = CLASS_ICONS[pred]

            st.markdown(f"""
            <div style="text-align: center;">
                <div class="prediction-badge" style="background-color: {color}; color: white;">
                    {icon} {name}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="confidence-container">', unsafe_allow_html=True)
            st.markdown('<div class="confidence-label">Confidence Score</div>', unsafe_allow_html=True)
            st.progress(conf, text=f"{conf*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("📊 View All Class Probabilities"):
                for i, prob in enumerate(all_probs):
                    st.write(f"**{CLASS_NAMES[i]}:** {prob*100:.2f}%")
                    st.progress(float(prob))

            st.markdown('</div>', unsafe_allow_html=True)

        # ---- Polyp detection ----
        if pred == 2 and yolo_loader[0] == "ultralytics":
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
                    for i, box in enumerate(boxes):
                        st.metric(f"Polyp #{i+1}", f"{box['conf']*100:.1f}%", delta="Detected")

                with detection_col2:
                    over = overlay_boxes_high_quality(img, boxes)
                    st.image(over, caption="Detected Polyps", use_container_width=True)

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

        # ---- UC severity ----
        if pred == 1 and ordinal_model:
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("### 📈 Ulcerative Colitis Severity Assessment")

            label, probs = predict_ordinal(ordinal_model, img)
            severity_names  = ["Remission (Mayo 0)", "Mild (Mayo 1)", "Moderate (Mayo 2)", "Severe (Mayo 3)"]
            severity_colors = ["#10b981", "#f59e0b", "#fb923c", "#ef4444"]

            severity_col1, severity_col2 = st.columns([1, 1])

            with severity_col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 1.5rem;">
                    <div class="severity-badge" style="background-color: {severity_colors[label]};">
                        {severity_names[label]}
                    </div>
                    <div style="margin-top: 1.5rem; font-size: 1.2rem; color: #1f2937; font-weight: 600;">
                        Mayo Grade: <strong style="color: {severity_colors[label]};">{label}/3</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with severity_col2:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown('<div class="info-box-title">Severity Class Probabilities</div>', unsafe_allow_html=True)

                for i, (p, sname) in enumerate(zip(probs, severity_names)):
                    prefix = "✓ " if i == label else "   "
                    st.markdown(f"""
                    <div style="margin: 0.75rem 0;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                            <span style="font-weight: {'700' if i == label else '600'}; color: {'#1f2937' if i == label else '#6b7280'};">
                                {prefix}{sname}
                            </span>
                            <span style="font-weight: 700; color: {severity_colors[i]};">{p*100:.2f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(float(p))

                st.markdown('</div>', unsafe_allow_html=True)

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
    Final Year Project 2026 • ResEffFusion Architecture with YOLO Detection
</div>
""", unsafe_allow_html=True)
