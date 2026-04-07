import os
import tempfile
import streamlit as st
import torch
import timm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
    menu_items={'About': "Deep Learning Framework for GI Tract Disorder Diagnosis"}
)

# ============================================================================
# CONSTANTS
# ============================================================================
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE     = 224
CLASS_NAMES  = ["Normal", "Ulcerative Colitis", "Polyps", "Esophagitis"]
CLASS_COLORS = {0: "#10b981", 1: "#f59e0b", 2: "#3b82f6", 3: "#8b5cf6"}
CLASS_ICONS  = {0: "✓", 1: "⚠", 2: "●", 3: "!"}

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

DRIVE_IDS = {
    "classifier": "1cvDPCfVBLWjCtx9mjz0KFWCwTwRs2atL",
    "ordinal":    "1Hvng1F6upAUjfsZe_sLpTmmH8lia04TI",
    "polyp":      "1xzGUJ1d9qDQKiodCzbWVOH07gNsffWnS",
}
MODEL_PATHS = {
    k: os.path.join(MODEL_DIR, f"best_{k}.pth" if k != "polyp" else "best.pt")
    for k in DRIVE_IDS
}

# ============================================================================
# CSS
# ============================================================================
st.markdown("""
<style>
    .block-container { padding-top:1rem; padding-bottom:0; max-width:100%; }
    .main-header {
        text-align:center; padding:2.5rem 1rem;
        background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
        border-radius:15px; margin-bottom:2rem;
    }
    .main-header h1 { font-size:2.5rem; font-weight:700; margin:0; color:white !important; }
    .main-header p  { font-size:1.1rem; margin:0.5rem 0 0; color:white !important; opacity:.95; }
    .result-card {
        background:white; border-radius:12px; padding:2rem;
        box-shadow:0 4px 6px rgba(0,0,0,.07);
        margin-bottom:1.5rem; border:1px solid #e5e7eb;
    }
    .image-card {
        background:white; border-radius:12px; padding:1.5rem;
        box-shadow:0 4px 6px rgba(0,0,0,.07);
        border:1px solid #e5e7eb; text-align:center; margin-bottom:1rem;
    }
    .image-card h3 { margin:0 0 1rem; color:#1f2937 !important; font-size:1.3rem; }
    .prediction-badge {
        display:inline-block; padding:1rem 2rem; border-radius:30px;
        font-size:1.5rem; font-weight:700; margin:1.5rem 0;
        box-shadow:0 4px 12px rgba(0,0,0,.2); color:white !important;
    }
    .confidence-container { margin:2rem 0; }
    .confidence-label { font-size:1rem; color:#374151 !important; margin-bottom:.75rem; font-weight:600; }
    .severity-badge {
        display:inline-block; padding:.75rem 1.5rem; border-radius:25px;
        font-size:1.2rem; font-weight:700; margin:1rem 0;
        color:white !important; box-shadow:0 2px 8px rgba(0,0,0,.15);
    }
    .detection-info {
        background:#dbeafe; border-left:5px solid #3b82f6;
        padding:1.25rem; border-radius:10px; margin:1.5rem 0;
    }
    .detection-title    { font-size:1.15rem; font-weight:700; color:#1e3a8a !important; margin-bottom:.5rem; }
    .detection-subtitle { font-size:.95rem; color:#1e40af !important; }
    .detection-info-warning { background:#fef3c7; border-left:5px solid #f59e0b; }
    .detection-info-warning .detection-title    { color:#78350f !important; }
    .detection-info-warning .detection-subtitle { color:#92400e !important; }
    .info-box {
        background:#f9fafb; border-radius:10px; padding:1.25rem;
        margin:1.5rem 0; border:1px solid #e5e7eb;
    }
    .info-box-title { font-weight:700; color:#1f2937 !important; margin-bottom:.75rem; font-size:1.1rem; }
    .divider {
        height:2px;
        background:linear-gradient(to right,transparent,#e5e7eb,transparent);
        margin:3rem 0;
    }
    .upload-section {
        background:linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);
        border-radius:12px; padding:3rem 2rem; text-align:center; margin:2rem 0;
    }
    .footer {
        text-align:center; padding:2rem 0; color:#6b7280 !important;
        font-size:.95rem; margin-top:3rem; border-top:1px solid #e5e7eb;
    }
    [data-testid="column"] { padding:.5rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class ResEffFusionClassifier(nn.Module):
    """4-class classifier: Normal / UC / Polyps / Esophagitis"""
    def __init__(self, num_classes=4, eff_weight=0.75):
        super().__init__()
        self.eff_weight = eff_weight
        self.res_weight = 1 - eff_weight
        self.eff        = timm.create_model("efficientnet_b4", pretrained=False, features_only=True)
        eff_dim         = self.eff.feature_info[-1]["num_chs"]
        self.res        = timm.create_model("resnet50", pretrained=False, features_only=True)
        res_dim         = self.res.feature_info[-1]["num_chs"]
        self.eff_proj   = nn.Conv2d(eff_dim, 1024, 1)
        self.res_proj   = nn.Conv2d(res_dim, 1024, 1)
        self.bn         = nn.BatchNorm2d(1024)
        self.relu       = nn.ReLU(inplace=False)
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        eff = self.eff(x)[-1]
        res = self.res(x)[-1]
        if eff.shape[2:] != res.shape[2:]:
            res = nn.functional.interpolate(res, size=eff.shape[2:], mode="bilinear", align_corners=False)
        fused = self.eff_weight * self.eff_proj(eff) + self.res_weight * self.res_proj(res)
        fused = self.relu(self.bn(fused))
        return self.classifier(self.pool(fused).flatten(1))


class ResEffFusionOrdinal(nn.Module):
    """Ordinal severity classifier — 4 outputs (Mayo 0-3)"""
    def __init__(self, num_classes=4, eff_weight=0.75):
        super().__init__()
        self.num_classes = num_classes
        self.eff_weight  = eff_weight
        self.res_weight  = 1 - eff_weight
        self.eff         = timm.create_model("efficientnet_b4", pretrained=False, features_only=True)
        eff_dim          = self.eff.feature_info[-1]["num_chs"]
        self.res         = timm.create_model("resnet50", pretrained=False, features_only=True)
        res_dim          = self.res.feature_info[-1]["num_chs"]
        self.eff_proj    = nn.Conv2d(eff_dim, 1024, 1)
        self.res_proj    = nn.Conv2d(res_dim, 1024, 1)
        self.bn          = nn.BatchNorm2d(1024)
        self.relu        = nn.ReLU(inplace=False)
        self.pool        = nn.AdaptiveAvgPool2d(1)
        self.classifier  = nn.Linear(1024, num_classes)   # 4 outputs — matches checkpoint

    def forward(self, x):
        eff = self.eff(x)[-1]
        res = self.res(x)[-1]
        if eff.shape[2:] != res.shape[2:]:
            res = nn.functional.interpolate(res, size=eff.shape[2:], mode="bilinear", align_corners=False)
        fused = self.eff_weight * self.eff_proj(eff) + self.res_weight * self.res_proj(res)
        fused = self.relu(self.bn(fused))
        return self.classifier(self.pool(fused).flatten(1))


# ============================================================================
# GRADCAM++
# Fix 1: Store .detach().clone() in hooks so values survive backward pass
# Fix 2: Switch model to train() before backward so BN hooks fire reliably,
#         then restore eval() — cached models stay in eval otherwise
# Fix 3: Guard against flat (all-zero) CAM → avoids solid blue from JET
# Fix 4: Remove hooks after use to prevent accumulation across calls
# ============================================================================

class GradCAMpp:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model       = model
        self._acts       = None
        self._grads      = None
        self._fwd_handle = target_layer.register_forward_hook(self._fwd_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, _m, _i, out):
        # .detach().clone() so it isn't freed when the graph is cleared
        self._acts = out.detach().clone()

    def _bwd_hook(self, _m, _i, grad_out):
        self._grads = grad_out[0].detach().clone()

    def remove(self):
        """Always call after generate() to prevent hook accumulation."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        # BatchNorm behaves differently in eval() vs train(); hooks on BN
        # layers need train() to capture proper intermediate activations.
        was_training = self.model.training
        self.model.train()

        with torch.enable_grad():
            self.model.zero_grad()
            logits = self.model(input_tensor.detach())
            score  = torch.softmax(logits, dim=1)[:, target_class]
            score.sum().backward()

        # Restore original mode
        if not was_training:
            self.model.eval()

        acts  = self._acts
        grads = self._grads

        # Safety: hooks didn't fire
        if acts is None or grads is None:
            return np.full((IMG_SIZE, IMG_SIZE), 0.0, dtype=np.float32)

        # GradCAM++ weighting
        g2      = grads ** 2
        g3      = grads ** 3
        sum_act = acts.sum(dim=(2, 3), keepdim=True)
        alpha   = g2 / (2.0 * g2 + sum_act * g3 + 1e-8)
        weights = (alpha * torch.relu(grads)).sum(dim=(2, 3))   # (1, C)

        cam = torch.relu((weights[:, :, None, None] * acts).sum(dim=1))
        cam = cam.squeeze().cpu().numpy().astype(np.float32)

        # Normalise — if cam is flat (all same value) return zeros, not blue
        cam_min, cam_max = cam.min(), cam.max()
        if (cam_max - cam_min) < 1e-6:
            cam = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        else:
            cam = (cam - cam_min) / (cam_max - cam_min)
            # Upsample with Pillow (no cv2)
            cam_pil = Image.fromarray((cam * 255).astype(np.uint8), mode="L")
            cam_pil = cam_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            cam     = np.array(cam_pil, dtype=np.float32) / 255.0

        return cam


# ============================================================================
# COLOURMAP + OVERLAY
# Fixed JET formula — the previous version produced solid blue for low values
# because the piecewise ranges were wrong.  This version matches matplotlib JET.
# ============================================================================

def _jet_colormap(gray: np.ndarray) -> np.ndarray:
    """
    Correct piecewise-linear JET colourmap (matches matplotlib).
    gray: float32 array in [0, 1], any shape
    returns: uint8 RGB array, same spatial shape + channel dim
    """
    r = np.clip(1.5 - np.abs(4.0 * gray - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * gray - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * gray - 1.0), 0.0, 1.0)
    # At gray=0: r=0, g=0, b=0.5  → dark blue  ✓
    # At gray=0.25: r=0, g=0.5, b=1 → cyan      ✓
    # At gray=0.5:  r=0.5, g=1, b=0.5 → green   ✓
    # At gray=0.75: r=1, g=0.5, b=0  → orange   ✓
    # At gray=1.0:  r=0.5, g=0, b=0  → dark red ✓
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def make_gradcam_overlay(cam: np.ndarray, original_pil: Image.Image, alpha: float = 0.5) -> Image.Image:
    orig  = np.array(original_pil.resize((IMG_SIZE, IMG_SIZE)).convert("RGB"), dtype=np.float32)
    heat  = _jet_colormap(cam).astype(np.float32)
    blend = ((1.0 - alpha) * orig + alpha * heat).clip(0, 255).astype(np.uint8)
    return Image.fromarray(blend)


# ============================================================================
# IMAGE TRANSFORM
# ============================================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

# ============================================================================
# HELPERS
# ============================================================================

def download_if_missing(file_id: str, out_path: str, desc: str) -> bool:
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        return True
    try:
        with st.spinner(f"Downloading {desc} model…"):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", out_path, quiet=False)
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
        probs = torch.softmax(model(x), dim=1)[0]
        pred  = int(probs.argmax())
    return pred, float(probs[pred]), probs.cpu().numpy()


def predict_ordinal(model, pil_img):
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    return int(np.argmax(probs)), probs


def run_yolo(yolo_loader, src_image: Image.Image, conf: float = 0.25) -> list:
    if yolo_loader[0] != "ultralytics":
        return []
    # Ultralytics expects BGR when given a numpy array (same as cv2 output).
    # Convert RGB PIL → numpy → flip channels to BGR to match original behaviour.
    img_rgb = np.array(src_image.resize((640, 640)).convert("RGB"))
    img_bgr = img_rgb[:, :, ::-1].copy()   # RGB→BGR, contiguous copy
    results = yolo_loader[1].predict(source=img_bgr, conf=conf, verbose=False)
    return [
        {"xyxy": np.array(b[:4]), "conf": b[4]}
        for b in results[0].boxes.data.tolist()
    ]


# ============================================================================
# YOLO BOUNDING BOX OVERLAY — pure Pillow, no cv2
# ============================================================================

def overlay_boxes_high_quality(image: Image.Image, boxes: list) -> Image.Image:
    img  = image.convert("RGBA")
    w, h = img.size

    ovl   = Image.new("RGBA", img.size, (0, 0, 0, 0))
    drw_o = ImageDraw.Draw(ovl)
    drw   = ImageDraw.Draw(img)

    lw        = max(3, min(w, h) // 180)
    corner_sz = max(18, min(w, h) // 18)
    font_size = max(15, min(w, h) // 28)

    font = ImageFont.load_default()
    for fpath in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            font = ImageFont.truetype(fpath, font_size)
            break
        except OSError:
            pass

    sx, sy = w / 640, h / 640

    for idx, box in enumerate(boxes):
        x1 = max(0,     int(box["xyxy"][0] * sx))
        y1 = max(0,     int(box["xyxy"][1] * sy))
        x2 = min(w - 1, int(box["xyxy"][2] * sx))
        y2 = min(h - 1, int(box["xyxy"][3] * sy))
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            continue
        cv = box["conf"]

        # Interior tint
        drw_o.rectangle([x1, y1, x2, y2], fill=(0, 230, 100, 22))

        # Box border
        for i in range(lw):
            drw.rectangle([x1+i, y1+i, x2-i, y2-i], outline=(0, 230, 100), width=1)

        # Corner L-brackets
        cs = min(corner_sz, (x2-x1)//3, (y2-y1)//3)
        ct = max(2, lw + 1)
        for rx1, ry1, rx2, ry2 in [
            (x1,    y1,    x1+cs, y1+ct),
            (x1,    y1,    x1+ct, y1+cs),
            (x2-cs, y1,    x2,    y1+ct),
            (x2-ct, y1,    x2,    y1+cs),
            (x1,    y2-ct, x1+cs, y2   ),
            (x1,    y2-cs, x1+ct, y2   ),
            (x2-cs, y2-ct, x2,    y2   ),
            (x2-ct, y2-cs, x2,    y2   ),
        ]:
            drw.rectangle([rx1, ry1, rx2, ry2], fill=(255, 255, 255))

        # Label pill
        label  = f"  Polyp {idx+1}   {cv*100:.0f}%  "
        bb     = drw.textbbox((0, 0), label, font=font)
        tw, th = bb[2]-bb[0], bb[3]-bb[1]
        px_p, py_p = 10, 6
        bar_h  = max(5, font_size // 4)
        pill_w = tw + 2 * px_p
        pill_h = th + 2 * py_p + bar_h + 4

        if y1 - pill_h - 6 >= 0:
            px, py = x1, y1 - pill_h - 6
        elif y2 + pill_h + 6 <= h:
            px, py = x1, y2 + 6
        else:
            px, py = x1 + 5, y1 + 5
        px = min(max(px, 0), w - pill_w - 2)

        drw_o.rectangle([px, py, px+pill_w, py+pill_h], fill=(0, 180, 80, 210))
        drw.text((px+px_p, py+py_p), label, fill=(255, 255, 255), font=font)

        # Confidence bar
        bar_y  = py + py_p + th + 4
        bx1    = px + px_p
        bx2    = px + pill_w - px_p
        bx_end = bx1 + int((bx2 - bx1) * cv)
        drw_o.rectangle([bx1, bar_y, bx2,    bar_y+bar_h], fill=(0, 100, 40, 200))
        if bx_end > bx1:
            drw_o.rectangle([bx1, bar_y, bx_end, bar_y+bar_h], fill=(160, 255, 160, 230))

    return Image.alpha_composite(img, ovl).convert("RGB")


# ============================================================================
# STREAMLIT APP
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
    "YOLO Detection Confidence", 0.05, 0.95, 0.25, 0.05,
    help="Minimum confidence for polyp detection"
)
show_gradcam = st.sidebar.checkbox(
    "Show GradCAM++ Heatmap", value=True,
    help="Visualise which image regions drive the model's decision"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Status")

dl_ok = {}
for k in DRIVE_IDS:
    ok = download_if_missing(DRIVE_IDS[k], MODEL_PATHS[k], k)
    dl_ok[k] = ok
    (st.sidebar.success if ok else st.sidebar.error)(f"{'✅' if ok else '❌'} {k.capitalize()}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💻 System Info")
st.sidebar.info(f"**Device:** {DEVICE.type.upper()}")
st.sidebar.info(f"**Image Size:** {IMG_SIZE}×{IMG_SIZE}")

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

# Info banner
st.markdown("""
<div class="info-box">
    <div class="info-box-title">📋 Supported Conditions</div>
    <div style="margin-top:.5rem;">
        <span style="color:#10b981;font-weight:600;">● Normal</span> •
        <span style="color:#f59e0b;font-weight:600;">● Ulcerative Colitis</span> •
        <span style="color:#3b82f6;font-weight:600;">● Polyps</span> •
        <span style="color:#8b5cf6;font-weight:600;">● Esophagitis</span>
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

if not uploaded:
    st.markdown("""
    <div class="upload-section">
        <h3 style="color:#374151;margin-top:0;">👆 Upload Images to Begin</h3>
        <p style="color:#6b7280;margin-bottom:0;">
            Upload colonoscopy images in JPG or PNG format for automated diagnosis
        </p>
    </div>""", unsafe_allow_html=True)

else:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("## 🔍 Analysis Results")

    progress_bar = st.progress(0)
    status_text  = st.empty()

    for idx, f in enumerate(uploaded):
        status_text.text(f"Processing image {idx+1} of {len(uploaded)}…")
        img = Image.open(f).convert("RGB")

        # ── Classification ─────────────────────────────────────────────────
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.markdown(f'<div class="image-card"><h3>📷 Image {idx+1}</h3>', unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            if not classification_model:
                st.error("❌ Classification model not loaded")
                st.markdown('</div>', unsafe_allow_html=True)
                continue

            pred, conf, all_probs = predict_class(classification_model, img)

            st.markdown(f"""
            <div style="text-align:center;">
                <div class="prediction-badge" style="background-color:{CLASS_COLORS[pred]};">
                    {CLASS_ICONS[pred]} {CLASS_NAMES[pred]}
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="confidence-container">'
                        '<div class="confidence-label">Confidence Score</div>',
                        unsafe_allow_html=True)
            st.progress(conf, text=f"{conf*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("📊 All Class Probabilities"):
                for i, p in enumerate(all_probs):
                    st.write(f"**{CLASS_NAMES[i]}:** {p*100:.2f}%")
                    st.progress(float(p))

            st.markdown('</div>', unsafe_allow_html=True)

        # ── GradCAM++ — classifier ──────────────────────────────────────────
        if show_gradcam and classification_model:
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("### 🧠 GradCAM++ — Classifier Attention")
            inp   = transform(img).unsqueeze(0).to(DEVICE)
            gcpp  = GradCAMpp(classification_model, classification_model.bn)
            cam   = gcpp.generate(inp, pred)
            gcpp.remove()   # clean up hooks

            gc1, gc2, gc3 = st.columns(3)
            with gc1:
                st.image(img.resize((IMG_SIZE, IMG_SIZE)), caption="Original",  use_container_width=True)
            with gc2:
                st.image(Image.fromarray(_jet_colormap(cam)), caption="Heatmap", use_container_width=True)
            with gc3:
                st.image(make_gradcam_overlay(cam, img),      caption="Overlay", use_container_width=True)

        # ── Polyp detection ─────────────────────────────────────────────────
        if pred == 2 and yolo_loader[0] == "ultralytics":
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("### 🎯 Polyp Detection Results")
            boxes = run_yolo(yolo_loader, img, conf_threshold)

            if boxes:
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.markdown(f"""
                    <div class="detection-info">
                        <div class="detection-title">✓ {len(boxes)} Polyp(s) Detected</div>
                        <div class="detection-subtitle">Threshold: {conf_threshold*100:.0f}%</div>
                    </div>""", unsafe_allow_html=True)
                    for i, b in enumerate(boxes):
                        st.metric(f"Polyp #{i+1}", f"{b['conf']*100:.1f}%", delta="Detected")
                with dc2:
                    over = overlay_boxes_high_quality(img, boxes)
                    st.image(over, caption="Detected Polyps", use_container_width=True)
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    over.save(tmp.name)
                    with open(tmp.name, "rb") as fp:
                        st.download_button(
                            "📥 Download Detection Image", fp,
                            file_name=f"polyp_detection_{idx+1}.png",
                            mime="image/png", key=f"dl_{idx}"
                        )
            else:
                st.markdown("""
                <div class="detection-info detection-info-warning">
                    <div class="detection-title">ℹ️ No polyps detected</div>
                    <div class="detection-subtitle">Try lowering the confidence threshold in the sidebar</div>
                </div>""", unsafe_allow_html=True)

        elif pred == 2:
            st.info("⚠️ YOLO model unavailable for polyp detection")

        # ── UC severity ─────────────────────────────────────────────────────
        if pred == 1 and ordinal_model:
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            st.markdown("### 📈 Ulcerative Colitis Severity Assessment")

            label, probs = predict_ordinal(ordinal_model, img)
            sev_names  = ["Remission (Mayo 0)", "Mild (Mayo 1)", "Moderate (Mayo 2)", "Severe (Mayo 3)"]
            sev_colors = ["#10b981", "#f59e0b", "#fb923c", "#ef4444"]

            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(f"""
                <div style="text-align:center;padding:1.5rem;">
                    <div class="severity-badge" style="background-color:{sev_colors[label]};">
                        {sev_names[label]}
                    </div>
                    <div style="margin-top:1.5rem;font-size:1.2rem;color:#1f2937;font-weight:600;">
                        Mayo Grade: <strong style="color:{sev_colors[label]};">{label}/3</strong>
                    </div>
                </div>""", unsafe_allow_html=True)

            with sc2:
                st.markdown('<div class="info-box"><div class="info-box-title">Severity Class Probabilities</div>',
                            unsafe_allow_html=True)
                for i, (p, sn) in enumerate(zip(probs, sev_names)):
                    prefix = "✓ " if i == label else "   "
                    st.markdown(f"""
                    <div style="margin:.75rem 0;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:.25rem;">
                            <span style="font-weight:{'700' if i==label else '600'};
                                         color:{'#1f2937' if i==label else '#6b7280'};">
                                {prefix}{sn}
                            </span>
                            <span style="font-weight:700;color:{sev_colors[i]};">{p*100:.2f}%</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                    st.progress(float(p))
                st.markdown('</div>', unsafe_allow_html=True)

            # GradCAM++ for ordinal model
            if show_gradcam:
                st.markdown("#### 🧠 GradCAM++ — Severity Model Attention")
                inp_ord = transform(img).unsqueeze(0).to(DEVICE)
                gcpp_ord = GradCAMpp(ordinal_model, ordinal_model.bn)
                cam_ord  = gcpp_ord.generate(inp_ord, label)
                gcpp_ord.remove()

                og1, og2, og3 = st.columns(3)
                with og1:
                    st.image(img.resize((IMG_SIZE, IMG_SIZE)), caption="Original", use_container_width=True)
                with og2:
                    st.image(Image.fromarray(_jet_colormap(cam_ord)), caption="Heatmap", use_container_width=True)
                with og3:
                    st.image(make_gradcam_overlay(cam_ord, img),      caption="Overlay", use_container_width=True)

        if idx < len(uploaded) - 1:
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        progress_bar.progress((idx + 1) / len(uploaded))

    status_text.text("✓ Analysis complete!")
    progress_bar.empty()
    status_text.empty()

# Footer
st.markdown("""
<div class="footer">
    <strong>Deep Learning Framework for GI Tract Disorder Diagnosis</strong><br>
    Final Year Project 2026 • ResEffFusion Architecture with YOLO Detection
</div>""", unsafe_allow_html=True)
