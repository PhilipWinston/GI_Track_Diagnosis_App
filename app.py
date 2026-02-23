# streamlit_app_final.py
"""
Streamlit app (final, minimal, uses only your 3 models and your format)
- Auto-downloads models from Google Drive (IDs hard-coded below)
- Runs classification (4 classes)
- If Polyps -> runs polyp detector (best.pt)
- If Ulcerative Colitis -> runs ordinal severity
- Score-CAM visualization (toggle)
"""

import os
import time
import tempfile
from typing import List, Tuple

import streamlit as st
import torch
import timm
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
import torch.nn as nn
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import gdown

# optional: ultralytics for best.pt - required if you want YOLO detection
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Colonoscopy Classifier (Final)", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)
IMG_SIZE = 224
DISPLAY_WIDTH = 380
CLASS_NAMES = ["Normal", "Ulcerative Colitis", "Polyps", "Esophagitis"]
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- YOUR GOOGLE DRIVE IDS (exactly as provided) ----------------
DRIVE_IDS = {
    "classifier": "1cvDPCfVBLWjCtx9mjz0KFWCwTwRs2atL",   # classifier .pth
    "ordinal":    "1Hvng1F6upAUjfsZe_sLpTmmH8lia04TI",   # ordinal .pth
    "polyp":      "1xzGUJ1d9qDQKiodCzbWVOH07gNsffWnS",   # best.pt
}

MODEL_PATHS = {
    "classifier": os.path.join(MODEL_DIR, "best_effresnetvit.pth"),
    "ordinal":   os.path.join(MODEL_DIR, "best_effresnetvit_ordinal.pth"),
    "polyp":     os.path.join(MODEL_DIR, "best.pt"),
}

# ---------------- Model classes (use same architecture you provided) ----------------
class EFFResNetViT(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.eff = timm.create_model("efficientnet_b4", pretrained=False, features_only=True)
        self.res = timm.create_model("resnet50", pretrained=False, features_only=True)
        eff_dim = self.eff.feature_info[-1]["num_chs"]
        res_dim = self.res.feature_info[-1]["num_chs"]
        self.fusion = nn.Conv2d(eff_dim + res_dim, 768, kernel_size=1)
        enc = nn.TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=3)
        self.classifier = nn.Sequential(nn.LayerNorm(768), nn.Dropout(0.4), nn.Linear(768, num_classes))

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
        enc = nn.TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072, dropout=0.1, batch_first=True)
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

# ---------------- transforms ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

# ---------------- Score-CAM (same approach) ----------------
class ScoreCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        target_layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.activations = out.detach()

    def generate(self, x, class_idx, max_maps=64):
        self.model.eval()
        with torch.no_grad():
            _ = self.model(x)
        if self.activations is None:
            return None
        maps = torch.relu(self.activations[0])  # C,H,W
        cam = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        num_maps = min(max_maps, maps.shape[0])
        for i in range(num_maps):
            m = maps[i]
            mmin, mmax = m.min(), m.max()
            if (mmax - mmin).abs() < 1e-6:
                continue
            m = (m - mmin) / (mmax - mmin + 1e-8)
            m_up = cv2.resize(m.cpu().numpy(), (IMG_SIZE, IMG_SIZE))
            m_tensor = torch.from_numpy(m_up).float().to(DEVICE)
            masked = x * m_tensor.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                out = self.model(masked)
                score = float(torch.softmax(out, dim=1)[0, class_idx].item())
            cam += score * m_up
        if cam.max() <= 0:
            return None
        return cam / (cam.max() + 1e-8)

# ---------------- download helper ----------------
def download_if_missing(file_id: str, out_path: str, desc: str):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        return True
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        with st.spinner(f"Downloading {desc}..."):
            gdown.download(url, out_path, quiet=False)
            time.sleep(0.5)
    except Exception as e:
        st.error(f"Download failed ({desc}): {e}")
        return False
    return os.path.exists(out_path) and os.path.getsize(out_path) > 1024

# ---------------- cached loaders ----------------
@st.cache_resource
def load_classification_model(path: str):
    m = EFFResNetViT(num_classes=4).to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    m.load_state_dict(state)
    m.eval()
    return m

@st.cache_resource
def load_ordinal_model(path: str):
    m = EFFResNetViTOrdinal(num_classes=4).to(DEVICE)
    state = torch.load(path, map_location=DEVICE)
    m.load_state_dict(state)
    m.eval()
    return m

@st.cache_resource
def load_yolo_model(path: str):
    if not HAS_ULTRALYTICS:
        return (None, None)
    try:
        y = YOLO(path)
        return ("ultralytics", y)
    except Exception:
        return (None, None)

# ---------------- prediction helpers ----------------
def predict_class(model, pil_img: Image.Image) -> Tuple[int, float, torch.Tensor]:
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(probs.argmax().item())
        conf = float(probs[pred].item())
    return pred, conf, x

def predict_ordinal(ordinal_model, pil_img: Image.Image) -> Tuple[int, np.ndarray]:
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = ordinal_model(x)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        label = int((probs > 0.5).sum())
    return label, probs

def run_yolo(yolo_loader, src_image: Image.Image, conf=0.25):
    img_bgr = cv2.cvtColor(np.array(src_image.resize((IMG_SIZE * 2, IMG_SIZE * 2))), cv2.COLOR_RGB2BGR)
    if yolo_loader[0] == "ultralytics":
        ymodel = yolo_loader[1]
        results = ymodel.predict(source=img_bgr, conf=conf, verbose=False)
        r = results[0]
        boxes = []
        try:
            for box in r.boxes:
                xyxy = box.xyxy.cpu().numpy()[0]
                confv = float(box.conf.cpu().numpy()[0])
                boxes.append({"xyxy": xyxy, "conf": confv})
        except Exception:
            for b in r.boxes.data.tolist():
                x1, y1, x2, y2, confv, cls = b
                boxes.append({"xyxy": np.array([x1, y1, x2, y2]), "conf": confv})
        return boxes
    return []

def overlay_boxes(image: Image.Image, boxes: List[dict]) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    sx = w / (IMG_SIZE * 2)
    sy = h / (IMG_SIZE * 2)
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        x1 = int(x1 * sx); x2 = int(x2 * sx)
        y1 = int(y1 * sy); y2 = int(y2 * sy)
        cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(arr, f"{b['conf']:.2f}", (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return Image.fromarray(arr)

# ---------------- UI (minimal) ----------------
st.title("Colonoscopy classifier — streamlined")
st.markdown("Uses only your provided models. Score-CAM toggle on right sidebar.")

# Sidebar options (minimal)
st.sidebar.header("Options")
show_scorecam = st.sidebar.checkbox("Show Score-CAM", value=True)
conf_threshold = st.sidebar.slider("YOLO conf threshold", 0.05, 0.95, 0.25)
st.sidebar.write(f"Device: {DEVICE}")

# Auto-download the exact three files you provided
dl_ok = {}
dl_ok["classifier"] = download_if_missing = download_if_missing = download_if_missing  # avoid lint error (no-op)
for k in DRIVE_IDS:
    ok = download_if_missing(DRIVE_IDS[k], MODEL_PATHS[k], desc=k)
    dl_ok[k] = ok
    if ok:
        st.sidebar.success(f"{k} present")
    else:
        st.sidebar.warning(f"{k} missing or failed to download")

# Load models if present
classification_model = None
ordinal_model = None
yolo_loader = (None, None)

if dl_ok.get("classifier") and os.path.exists(MODEL_PATHS["classifier"]):
    try:
        classification_model = load_classification_model(MODEL_PATHS["classifier"])
    except Exception as e:
        st.error(f"Failed to load classifier: {e}")

if dl_ok.get("ordinal") and os.path.exists(MODEL_PATHS["ordinal"]):
    try:
        ordinal_model = load_ordinal_model(MODEL_PATHS["ordinal"])
    except Exception as e:
        st.warning(f"Ordinal load failed: {e}")

if dl_ok.get("polyp") and os.path.exists(MODEL_PATHS["polyp"]) and HAS_ULTRALYTICS:
    try:
        yolo_loader = load_yolo_model(MODEL_PATHS["polyp"])
    except Exception as e:
        st.warning(f"YOLO load failed: {e}")
elif dl_ok.get("polyp") and not HAS_ULTRALYTICS:
    st.sidebar.info("YOLO present but ultralytics not installed; polyp detection disabled.")

# Image uploader
uploaded = st.file_uploader("Upload image(s) (jpg/png). Multiple allowed.", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded:
    progress = st.progress(0)
    total = len(uploaded)
    for idx, f in enumerate(uploaded):
        try:
            img = Image.open(f).convert("RGB")
        except Exception as e:
            st.error(f"Cannot open image: {e}")
            continue

        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.image(img, caption=f"Original #{idx+1}", width=DISPLAY_WIDTH)

        # classification
        if classification_model is None:
            with col2:
                st.warning("Classifier not loaded.")
            continue

        pred, conf, x = predict_class(classification_model, img)
        with col2:
            st.markdown(f"**Prediction:** {CLASS_NAMES[pred]}")
            st.markdown(f"**Confidence:** {conf*100:.2f}%")

            # polyp detection
            if CLASS_NAMES[pred].lower().startswith("poly"):
                if yolo_loader[0] == "ultralytics" and yolo_loader[1] is not None:
                    boxes = run_yolo(yolo_loader, img, conf=conf_threshold)
                    if boxes:
                        over = overlay_boxes(img, boxes)
                        st.image(over, caption="Detections overlay", width=DISPLAY_WIDTH)
                        # download button
                        tmpf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                        over.save(tmpf.name)
                        st.download_button("Download detections", data=open(tmpf.name, "rb"), file_name=f"detections_{idx+1}.png")
                    else:
                        st.info("No detections found.")
                else:
                    st.info("Polyp detector not available (ultralytics not installed or load failed).")

            # ulcer ordinal severity
            if CLASS_NAMES[pred].lower().startswith("ulcer"):
                if ordinal_model is not None:
                    label, probs = predict_ordinal(ordinal_model, img)
                    st.markdown(f"**Ordinal severity (0..3):** {label}")
                    st.markdown(f"**Sigmoid outputs:** `{np.round(probs, 3)}`")
                else:
                    st.info("Ordinal model not available.")

            # Score-CAM
            if show_scorecam:
                try:
                    sc = ScoreCAM(classification_model, classification_model.eff.blocks[4])
                    cam = sc.generate(x, pred)
                    if cam is None:
                        st.info("Score-CAM had no activation.")
                    else:
                        heat = cv2.applyColorMap((cam * 255).astype("uint8"), cv2.COLORMAP_JET)
                        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
                        img_resized = np.array(img.resize((IMG_SIZE, IMG_SIZE))).astype("uint8")
                        overlay = (0.6 * img_resized + 0.4 * heat).astype("uint8")
                        st.image(heat, caption="Score-CAM heatmap", width=DISPLAY_WIDTH)
                        st.image(overlay, caption="Score-CAM overlay", width=DISPLAY_WIDTH)
                        tmpf = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                        Image.fromarray(overlay).save(tmpf.name)
                        st.download_button("Download Score-CAM overlay", data=open(tmpf.name, "rb"), file_name=f"scorecam_{idx+1}.png")
                except Exception as e:
                    st.warning(f"Score-CAM failed: {e}")

        progress.progress(int(((idx + 1) / total) * 100))
    progress.empty()
else:
    st.info("Upload images to run the pipeline.")

st.markdown("---")
st.caption("This app uses only the three models you provided (hard-coded). If polyp detection is not running, install the 'ultralytics' package on the server.")