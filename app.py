
import os
import tempfile
from typing import Optional, Tuple, Dict, Any, List

import cv2
import gdown
import numpy as np
import streamlit as st
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="GI Tract Diagnosis System",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Deep Learning Framework for GI Tract Disorder Diagnosis"
    },
)

# ============================================================================
# CONSTANTS
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
DISPLAY_WIDTH = 500

CLASS_NAMES = ["Normal", "Ulcerative Colitis", "Polyps", "Esophagitis"]
CLASS_COLORS = {
    0: "#10b981",
    1: "#f59e0b",
    2: "#3b82f6",
    3: "#8b5cf6",
}
CLASS_ICONS = {
    0: "✓",
    1: "⚠",
    2: "●",
    3: "!",
}

SEVERITY_NAMES = ["Remission", "Mild", "Moderate", "Severe"]
SEVERITY_COLORS = ["#10b981", "#f59e0b", "#fb923c", "#ef4444"]

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Use the new model filenames from your updated training scripts.
MODEL_PATHS = {
    "classifier": os.path.join(MODEL_DIR, "best_reseff_fusion_masked.pth"),
    "ordinal": os.path.join(MODEL_DIR, "best_reseff_fusion_masked_ordinal.pth"),
    "polyp": os.path.join(MODEL_DIR, "best.pt"),
}

# Optional Google Drive IDs if you still want auto-download.
# Leave as None if you only want local loading.
DRIVE_IDS = {
    "classifier": None,
    "ordinal": None,
    "polyp": None,
}

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# ============================================================================
# MODEL CLASSES
# ============================================================================
class ResEffFusion(nn.Module):
    """
    New classification model:
    EfficientNet-B4 + ResNet50 fusion with weighted feature blending.
    """
    def __init__(self, num_classes: int, eff_weight: float = 0.75):
        super().__init__()
        self.eff_weight = eff_weight
        self.res_weight = 1 - eff_weight

        self.eff = timm.create_model("efficientnet_b4", pretrained=False, features_only=True)
        eff_dim = self.eff.feature_info[-1]["num_chs"]

        self.res = timm.create_model("resnet50", pretrained=False, features_only=True)
        res_dim = self.res.feature_info[-1]["num_chs"]

        self.eff_proj = nn.Conv2d(eff_dim, 1024, 1)
        self.res_proj = nn.Conv2d(res_dim, 1024, 1)

        self.bn = nn.BatchNorm2d(1024)
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
    """
    New ordinal/CORAL style model.
    Returns K-1 logits for K classes.
    """
    def __init__(self, num_classes: int, eff_weight: float = 0.75):
        super().__init__()
        self.eff_weight = eff_weight
        self.res_weight = 1 - eff_weight

        self.eff = timm.create_model("efficientnet_b4", pretrained=False, features_only=True)
        eff_dim = self.eff.feature_info[-1]["num_chs"]

        self.res = timm.create_model("resnet50", pretrained=False, features_only=True)
        res_dim = self.res.feature_info[-1]["num_chs"]

        self.eff_proj = nn.Conv2d(eff_dim, 1024, 1)
        self.res_proj = nn.Conv2d(res_dim, 1024, 1)

        self.bn = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
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
# TRANSFORMS
# ============================================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

# ============================================================================
# HELPERS
# ============================================================================
def download_if_missing(file_id: Optional[str], out_path: str, desc: str) -> bool:
    """Download a model from Google Drive if not present."""
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        return True

    if not file_id:
        return False

    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        with st.spinner(f"Downloading {desc} model..."):
            gdown.download(url, out_path, quiet=False)
        return os.path.exists(out_path) and os.path.getsize(out_path) > 1024
    except Exception:
        return False


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    """Handle plain state_dict or checkpoint dictionaries."""
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove common prefixes such as 'module.' from DataParallel checkpoints."""
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module.") :]
        if new_k.startswith("model."):
            new_k = new_k[len("model.") :]
        cleaned[new_k] = v
    return cleaned


def load_checkpoint_into_model(model: nn.Module, path: str) -> nn.Module:
    ckpt = torch.load(path, map_location=DEVICE)
    state_dict = extract_state_dict(ckpt)
    state_dict = clean_state_dict_keys(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@st.cache_resource
def load_classification_model(path: str):
    model = ResEffFusion(num_classes=4).to(DEVICE)
    return load_checkpoint_into_model(model, path)


@st.cache_resource
def load_ordinal_model(path: str):
    model = ResEffFusionOrdinal(num_classes=4).to(DEVICE)
    return load_checkpoint_into_model(model, path)


@st.cache_resource
def load_yolo_model(path: str):
    if not HAS_ULTRALYTICS:
        return (None, None)
    try:
        return ("ultralytics", YOLO(path))
    except Exception:
        return (None, None)


def predict_class(model, pil_img):
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(probs.argmax())
        conf = float(probs[pred])
    return pred, conf, probs.cpu().numpy()


def coral_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert K-1 ordinal logits to K class probabilities.
    """
    sig = torch.sigmoid(logits)  # (B, K-1)
    k_minus_1 = logits.size(1)
    probs = []

    probs.append((1 - sig[:, 0]).unsqueeze(1))
    for r in range(1, k_minus_1):
        probs.append((sig[:, r - 1] - sig[:, r]).unsqueeze(1))
    probs.append(sig[:, k_minus_1 - 1].unsqueeze(1))

    probs = torch.cat(probs, dim=1)
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-12)
    return probs


def predict_ordinal(ordinal_model, pil_img):
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = ordinal_model(x)
        probs = coral_logits_to_probs(logits)[0].cpu().numpy()

    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / (probs.sum() + 1e-12)
    label = int(np.argmax(probs))
    return label, probs


class GradCAMpp:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        self.fwd_handle = target_layer.register_forward_hook(self.forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        self.activations = out

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def remove(self):
        if self.fwd_handle is not None:
            self.fwd_handle.remove()
        if self.bwd_handle is not None:
            self.bwd_handle.remove()

    def generate(self, input_tensor, target_class, ordinal: bool = False):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)

        if ordinal:
            probs = coral_logits_to_probs(logits)
            score = probs[:, target_class].sum()
        else:
            score = logits[:, target_class].sum()

        score.backward(retain_graph=True)

        activations = self.activations
        gradients = self.gradients

        if activations is None or gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients.")

        grads_power_2 = gradients ** 2
        grads_power_3 = gradients ** 3
        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)

        alpha = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + 1e-8)
        positive_grad = torch.relu(gradients)
        weights = torch.sum(alpha * positive_grad, dim=(2, 3))

        cam = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * activations, dim=1)
        cam = torch.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cv2.resize(cam, (IMG_SIZE, IMG_SIZE))


def generate_gradcam_overlay(model, input_tensor, pil_img, pred_class, ordinal: bool = False):
    campp = GradCAMpp(model, model.bn)
    try:
        cam = campp.generate(input_tensor, pred_class, ordinal=ordinal)
    finally:
        campp.remove()

    img_resized = cv2.resize(np.array(pil_img.convert("RGB")), (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
    return img_resized, heatmap, overlay


def run_yolo(yolo_loader, src_image, conf=0.25):
    """Run YOLO polyp detection"""
    if yolo_loader[0] != "ultralytics":
        return []

    img_bgr = cv2.cvtColor(
        np.array(src_image.resize((640, 640))),
        cv2.COLOR_RGB2BGR,
    )

    results = yolo_loader[1].predict(source=img_bgr, conf=conf, verbose=False)
    r = results[0]

    boxes = []
    for b in r.boxes.data.tolist():
        x1, y1, x2, y2, confv, cls = b
        boxes.append({
            "xyxy": np.array([x1, y1, x2, y2]),
            "conf": confv,
        })

    return boxes


def overlay_boxes_high_quality(image, boxes):
    """Overlay detection boxes with high quality rendering"""
    img_pil = image.convert("RGB")
    w, h = img_pil.size

    draw = ImageDraw.Draw(img_pil)
    font_size = max(14, min(w, h) // 35)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except Exception:
            font = ImageFont.load_default()

    sx = w / 640
    sy = h / 640

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box["xyxy"]
        x1 = max(0, int(x1 * sx))
        y1 = max(0, int(y1 * sy))
        x2 = min(w, int(x2 * sx))
        y2 = min(h, int(y2 * sy))

        line_width = max(3, min(w, h) // 200)
        for i in range(line_width):
            draw.rectangle(
                [x1 + i, y1 + i, x2 - i, y2 - i],
                outline=(0, 255, 0),
                width=1,
            )

        label = f"Polyp {idx + 1} - {box['conf']*100:.0f}%"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        pad = 6
        label_w = text_w + 2 * pad
        label_h = text_h + 2 * pad

        if y1 - label_h - 5 > 0:
            label_x = x1
            label_y = y1 - label_h - 5
        elif y2 + label_h + 5 < h:
            label_x = x1
            label_y = y2 + 5
        else:
            label_x = x1 + 5
            label_y = y1 + 5

        if label_x + label_w > w:
            label_x = w - label_w - 5
        label_x = max(5, label_x)

        draw.rectangle(
            [label_x, label_y, label_x + label_w, label_y + label_h],
            fill=(0, 200, 0),
            outline=(0, 150, 0),
            width=2,
        )
        draw.text(
            (label_x + pad, label_y + pad),
            label,
            fill=(255, 255, 255),
            font=font,
        )

    return img_pil


# ============================================================================
# MAIN APPLICATION
# ============================================================================
st.markdown(
    """
<div class="main-header">
    <h1>🔬 GI Tract Diagnosis System</h1>
    <p>Deep Learning Framework for Gastrointestinal Disorder Detection</p>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.header("⚙️ Configuration")

conf_threshold = st.sidebar.slider(
    "YOLO Detection Confidence",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05,
    help="Minimum confidence threshold for polyp detection",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Status")

dl_ok = {}
for k, p in MODEL_PATHS.items():
    if DRIVE_IDS.get(k):
        ok = download_if_missing(DRIVE_IDS[k], p, k)
    else:
        ok = os.path.exists(p) and os.path.getsize(p) > 1024
    dl_ok[k] = ok
    if ok:
        st.sidebar.success(f"✅ {k.capitalize()} Model")
    else:
        st.sidebar.error(f"❌ {k.capitalize()} Model")

classification_model = None
ordinal_model = None
yolo_loader = (None, None)

if dl_ok.get("classifier"):
    classification_model = load_classification_model(MODEL_PATHS["classifier"])

if dl_ok.get("ordinal"):
    ordinal_model = load_ordinal_model(MODEL_PATHS["ordinal"])

if dl_ok.get("polyp") and HAS_ULTRALYTICS:
    yolo_loader = load_yolo_model(MODEL_PATHS["polyp"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 💻 System Info")
st.sidebar.info(f"**Device:** {DEVICE.type.upper()}")
st.sidebar.info(f"**Image Size:** {IMG_SIZE}×{IMG_SIZE}")

st.markdown(
    """
<div class="info-box">
    <div class="info-box-title">📋 Supported Conditions</div>
    <div style="margin-top: 0.5rem;">
        <span style="color: #10b981; font-weight: 600;">● Normal</span> •
        <span style="color: #f59e0b; font-weight: 600;">● Ulcerative Colitis</span> •
        <span style="color: #3b82f6; font-weight: 600;">● Polyps</span> •
        <span style="color: #8b5cf6; font-weight: 600;">● Esophagitis</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "📤 Upload Colonoscopy Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Upload one or more colonoscopy images for analysis",
)

if uploaded:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("## 🔍 Analysis Results")

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, f in enumerate(uploaded):
        status_text.text(f"Processing image {idx + 1} of {len(uploaded)}...")

        img = Image.open(f).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)

        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.markdown(
                f"""
            <div class="image-card">
                <h3>📷 Image {idx + 1}</h3>
            """,
                unsafe_allow_html=True,
            )
            st.image(img, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)

            if not classification_model:
                st.error("❌ Classification model not loaded")
                st.markdown("</div>", unsafe_allow_html=True)
                continue

            pred, conf, all_probs = predict_class(classification_model, img)
            color = CLASS_COLORS[pred]
            name = CLASS_NAMES[pred]
            icon = CLASS_ICONS[pred]

            st.markdown(
                f"""
            <div style="text-align: center;">
                <div class="prediction-badge" style="background-color: {color}; color: white;">
                    {icon} {name}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            st.markdown('<div class="confidence-container">', unsafe_allow_html=True)
            st.markdown('<div class="confidence-label">Confidence Score</div>', unsafe_allow_html=True)
            st.progress(conf, text=f"{conf*100:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("📊 View All Class Probabilities"):
                for i, prob in enumerate(all_probs):
                    st.write(f"**{CLASS_NAMES[i]}:** {prob*100:.2f}%")
                    st.progress(float(prob))

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("### 🧠 GradCAM++ Explanation")

        try:
            img_resized, heatmap, overlay = generate_gradcam_overlay(
                classification_model,
                input_tensor,
                img,
                pred,
                ordinal=False,
            )

            cam_col1, cam_col2, cam_col3 = st.columns(3)
            with cam_col1:
                st.image(img_resized, caption="Original", use_container_width=True)
            with cam_col2:
                st.image(heatmap, caption="GradCAM++ Heatmap", use_container_width=True)
            with cam_col3:
                st.image(overlay, caption="Overlay", use_container_width=True)
        except Exception as e:
            st.warning(f"GradCAM could not be generated for this image: {e}")

        if pred == 2 and yolo_loader[0] == "ultralytics":
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("### 🎯 Polyp Detection Results")

            boxes = run_yolo(yolo_loader, img, conf_threshold)

            if boxes:
                detection_col1, detection_col2 = st.columns([1, 1])

                with detection_col1:
                    st.markdown(
                        f"""
                    <div class="detection-info">
                        <div class="detection-title">✓ {len(boxes)} Polyp(s) Detected</div>
                        <div class="detection-subtitle">Confidence threshold: {conf_threshold*100:.0f}%</div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    for i, box in enumerate(boxes):
                        st.metric(
                            f"Polyp #{i + 1}",
                            f"{box['conf']*100:.1f}%",
                            delta="Detected",
                        )

                with detection_col2:
                    over = overlay_boxes_high_quality(img, boxes)
                    st.image(over, caption="Detected Polyps", use_container_width=True)

                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    over.save(tmp.name, quality=95)
                    with open(tmp.name, "rb") as file:
                        st.download_button(
                            label="📥 Download Detection Image",
                            data=file,
                            file_name=f"polyp_detection_{idx + 1}.png",
                            mime="image/png",
                            key=f"download_detection_{idx}",
                        )
            else:
                st.markdown(
                    """
                <div class="detection-info detection-info-warning">
                    <div class="detection-title">ℹ️ No polyps detected</div>
                    <div class="detection-subtitle">Try adjusting the confidence threshold in the sidebar</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        elif pred == 2:
            st.info("⚠️ YOLO model unavailable for polyp detection")

        if pred == 1 and ordinal_model:
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("### 📈 Ulcerative Colitis Severity Assessment")

            label, probs = predict_ordinal(ordinal_model, img)

            severity_col1, severity_col2 = st.columns([1, 1])

            with severity_col1:
                st.markdown(
                    f"""
                <div style="text-align: center; padding: 1.5rem;">
                    <div class="severity-badge" style="background-color: {SEVERITY_COLORS[label]};">
                        {SEVERITY_NAMES[label]}
                    </div>
                    <div style="margin-top: 1.5rem; font-size: 1.2rem; color: #1f2937; font-weight: 600;">
                        Severity Grade: <strong style="color: {SEVERITY_COLORS[label]};">{label}/3</strong>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            with severity_col2:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown('<div class="info-box-title">Severity Class Probabilities</div>', unsafe_allow_html=True)

                for i, (p, name_) in enumerate(zip(probs, SEVERITY_NAMES)):
                    prefix = "✓ " if i == label else "   "
                    st.markdown(
                        f"""
                    <div style="margin: 0.75rem 0;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                            <span style="font-weight: {'700' if i == label else '600'}; color: {'#1f2937' if i == label else '#6b7280'};">
                                {prefix}{name_} (Grade {i})
                            </span>
                            <span style="font-weight: 700; color: {SEVERITY_COLORS[i]};">{p*100:.2f}%</span>
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                    st.progress(float(p))

                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("### 🧠 Ordinal GradCAM++ Explanation")
            try:
                img_resized, heatmap, overlay = generate_gradcam_overlay(
                    ordinal_model,
                    input_tensor,
                    img,
                    label,
                    ordinal=True,
                )

                cam_col1, cam_col2, cam_col3 = st.columns(3)
                with cam_col1:
                    st.image(img_resized, caption="Original", use_container_width=True)
                with cam_col2:
                    st.image(heatmap, caption="GradCAM++ Heatmap", use_container_width=True)
                with cam_col3:
                    st.image(overlay, caption="Overlay", use_container_width=True)
            except Exception as e:
                st.warning(f"Ordinal GradCAM could not be generated: {e}")

        if idx < len(uploaded) - 1:
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        progress_bar.progress((idx + 1) / len(uploaded))

    status_text.text("✓ Analysis complete!")
    progress_bar.empty()
    status_text.empty()

else:
    st.markdown(
        """
    <div class="upload-section">
        <h3 style="color: #374151; margin-top: 0;">👆 Upload Images to Begin</h3>
        <p style="color: #6b7280; margin-bottom: 0;">
            Upload colonoscopy images in JPG or PNG format for automated diagnosis
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="footer">
    <strong>Deep Learning Framework for GI Tract Disorder Diagnosis</strong><br>
    Final Year Project 2026 • ResEffFusion Architecture with YOLO Detection
</div>
""",
    unsafe_allow_html=True,
)
