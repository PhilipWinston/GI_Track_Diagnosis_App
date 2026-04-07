import os
from pathlib import Path

import gdown
import numpy as np
import streamlit as st
import timm
import torch
import torch.nn as nn
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False


# ============================================================================
# CONFIG
# ============================================================================
st.set_page_config(page_title="GI Tract Diagnosis System", layout="wide", initial_sidebar_state="expanded")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

CLASS_NAMES = ["Normal", "Ulcerative Colitis", "Polyps", "Esophagitis"]
SEVERITY_NAMES = ["Mayo 0", "Mayo 1", "Mayo 2", "Mayo 3"]

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DRIVE_IDS = {
    "classifier": "1aZke_47izApUtev2-Jlr1j4ZC84DC1i1",
    "ordinal": "1Q74a7he0LnLfDJEN90YLpMhzJxI0wKa2",
    "polyp": "1xzGUJ1d9qDQKiodCzbWVOH07gNsffWnS",
}

MODEL_PATHS = {
    "classifier": str(MODEL_DIR / "best_classifier.pth"),
    "ordinal": str(MODEL_DIR / "best_ordinal.pth"),
    "polyp": str(MODEL_DIR / "best.pt"),
}

transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ]
)


# ============================================================================
# SIMPLE UI
# ============================================================================
st.title("GI Tract Diagnosis System")
st.caption("Simple working app for classification, UC severity, and YOLOv11 polyp detection")

st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("YOLO confidence", 0.05, 0.95, 0.25, 0.05)
show_probs = st.sidebar.checkbox("Show probabilities", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Status")


# ============================================================================
# MODEL
# ============================================================================
class ResEffFusion(nn.Module):
    def __init__(self, num_outputs=4, eff_weight=0.75):
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
        self.classifier = nn.Linear(1024, num_outputs)

    def forward(self, x):
        eff = self.eff(x)[-1]
        res = self.res(x)[-1]
        if eff.shape[2:] != res.shape[2:]:
            res = nn.functional.interpolate(res, size=eff.shape[2:], mode="bilinear", align_corners=False)
        fused = self.eff_weight * self.eff_proj(eff) + self.res_weight * self.res_proj(res)
        fused = self.relu(self.bn(fused))
        return self.classifier(self.pool(fused).flatten(1))


# ============================================================================
# CHECKPOINT HELPERS
# ============================================================================
def safe_torch_load(path: str):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=DEVICE)
    except Exception:
        return torch.load(path, map_location=DEVICE)


def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "module"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                ckpt_obj = ckpt_obj[key]
                break

    if not isinstance(ckpt_obj, dict):
        raise ValueError("Unsupported checkpoint format")

    cleaned = {}
    for k, v in ckpt_obj.items():
        cleaned[k.replace("module.", "")] = v
    return cleaned


def infer_num_outputs(state_dict: dict, default_outputs: int):
    for key in ("classifier.weight", "head.weight", "fc.weight"):
        if key in state_dict and hasattr(state_dict[key], "shape"):
            return int(state_dict[key].shape[0])
    return default_outputs


@st.cache_resource
def load_torch_model(path: str, default_outputs: int):
    ckpt = safe_torch_load(path)
    state_dict = extract_state_dict(ckpt)
    num_outputs = infer_num_outputs(state_dict, default_outputs)

    model = ResEffFusion(num_outputs=num_outputs).to(DEVICE)
    model_sd = model.state_dict()

    compatible = {}
    for k, v in state_dict.items():
        if k in model_sd and hasattr(v, "shape") and v.shape == model_sd[k].shape:
            compatible[k] = v

    model_sd.update(compatible)
    model.load_state_dict(model_sd, strict=False)
    model.eval()
    return model, num_outputs


@st.cache_resource
def load_yolo_model(path: str):
    if not HAS_ULTRALYTICS:
        return None
    try:
        return YOLO(path)
    except Exception:
        return None


# ============================================================================
# PREDICTION HELPERS
# ============================================================================
def coral_logits_to_probs(logits: torch.Tensor) -> np.ndarray:
    sig = torch.sigmoid(logits)
    k = logits.size(1) + 1
    parts = [(1 - sig[:, 0]).unsqueeze(1)]
    for r in range(1, k - 1):
        parts.append((sig[:, r - 1] - sig[:, r]).unsqueeze(1))
    parts.append(sig[:, k - 2].unsqueeze(1))
    probs = torch.cat(parts, dim=1)
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-12)
    return probs.squeeze(0).detach().cpu().numpy()


def predict_classification(model, image: Image.Image):
    x = transform(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    pred = int(np.argmax(probs))
    return pred, float(probs[pred]), probs


def predict_ordinal(model, image: Image.Image):
    x = transform(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(x)
        if logits.size(1) == 3:
            probs = coral_logits_to_probs(logits)
        elif logits.size(1) == 4:
            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        else:
            raise ValueError(f"Unexpected ordinal head size: {logits.size(1)}")
    probs = np.clip(probs, 0, 1)
    probs = probs / (probs.sum() + 1e-12)
    return int(np.argmax(probs)), probs


def yolo_predict_and_annotate(yolo_model, image: Image.Image, conf: float):
    if yolo_model is None:
        return None, []

    img_np = np.array(image.convert("RGB"))
    results = yolo_model.predict(source=img_np, conf=conf, imgsz=640, verbose=False)

    if not results:
        return image.convert("RGB"), []

    r0 = results[0]
    annotated = r0.plot()  # BGR numpy array
    annotated = Image.fromarray(annotated[:, :, ::-1].copy())

    detections = []
    if getattr(r0, "boxes", None) is not None and len(r0.boxes) > 0:
        xyxy = r0.boxes.xyxy.cpu().numpy()
        confs = r0.boxes.conf.cpu().numpy()
        cls_ids = r0.boxes.cls.cpu().numpy().astype(int)
        names = r0.names if hasattr(r0, "names") else {}

        for i in range(len(xyxy)):
            detections.append(
                {
                    "xyxy": xyxy[i],
                    "conf": float(confs[i]),
                    "cls_id": int(cls_ids[i]),
                    "cls_name": names.get(int(cls_ids[i]), f"class_{int(cls_ids[i])}"),
                }
            )

    return annotated, detections


# ============================================================================
# LOAD MODELS
# ============================================================================
def download_if_missing(file_id: str, out_path: str) -> bool:
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        return True
    try:
        gdown.download(f"https://drive.google.com/uc?id={file_id}", out_path, quiet=False)
        return os.path.exists(out_path) and os.path.getsize(out_path) > 1024
    except Exception:
        return False


dl_ok = {}
for name, file_id in DRIVE_IDS.items():
    ok = download_if_missing(file_id, MODEL_PATHS[name])
    dl_ok[name] = ok
    st.sidebar.write(f"{'✅' if ok else '❌'} {name}")

st.sidebar.markdown("---")
st.sidebar.write(f"Device: {DEVICE.type.upper()}")

classifier_model = None
ordinal_model = None
yolo_model = None

try:
    if dl_ok.get("classifier") and os.path.exists(MODEL_PATHS["classifier"]):
        classifier_model, _ = load_torch_model(MODEL_PATHS["classifier"], default_outputs=4)
except Exception:
    classifier_model = None

try:
    if dl_ok.get("ordinal") and os.path.exists(MODEL_PATHS["ordinal"]):
        ordinal_model, _ = load_torch_model(MODEL_PATHS["ordinal"], default_outputs=3)
except Exception:
    ordinal_model = None

try:
    if dl_ok.get("polyp") and os.path.exists(MODEL_PATHS["polyp"]) and HAS_ULTRALYTICS:
        yolo_model = load_yolo_model(MODEL_PATHS["polyp"])
except Exception:
    yolo_model = None

if not HAS_ULTRALYTICS:
    st.sidebar.error("ultralytics not installed")
elif yolo_model is None:
    st.sidebar.warning("YOLO model not loaded")
else:
    st.sidebar.success("YOLO loaded")


# ============================================================================
# UPLOAD
# ============================================================================
uploaded_files = st.file_uploader(
    "Upload colonoscopy image(s)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more images to start.")
    st.stop()


# ============================================================================
# APP CONTENT
# ============================================================================
for idx, f in enumerate(uploaded_files):
    image = Image.open(f).convert("RGB")

    st.divider()
    st.subheader(f"Image {idx + 1}")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.image(image, use_container_width=True)

    with c2:
        if classifier_model is None:
            st.error("Classification model not loaded.")
        else:
            pred, conf, probs = predict_classification(classifier_model, image)
            st.write(f"**Prediction:** {CLASS_NAMES[pred]}")
            st.write(f"**Confidence:** {conf * 100:.2f}%")

            if show_probs:
                for i, p in enumerate(probs):
                    st.write(f"{CLASS_NAMES[i]}: {p * 100:.2f}%")
                    st.progress(float(p))

            if pred == 2:
                st.markdown("### YOLOv11 Polyp Detection")
                if yolo_model is None:
                    st.error("YOLO model not available.")
                else:
                    annotated, detections = yolo_predict_and_annotate(yolo_model, image, conf_threshold)
                    st.image(annotated, caption="YOLOv11 bounding boxes", use_container_width=True)

                    if detections:
                        st.success(f"{len(detections)} polyp(s) detected")
                        for i, d in enumerate(detections, 1):
                            x1, y1, x2, y2 = d["xyxy"]
                            st.write(
                                f"**Box {i}** | {d['cls_name']} | conf: {d['conf']:.2f} | "
                                f"xyxy: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
                            )
                    else:
                        st.info("No polyp detected.")

            if pred == 1 and ordinal_model is not None:
                st.markdown("### Ulcerative Colitis Severity")
                try:
                    severity_label, severity_probs = predict_ordinal(ordinal_model, image)
                    st.write(f"**Severity:** {SEVERITY_NAMES[severity_label]}")
                    st.write(f"**Mayo grade:** {severity_label}/3")

                    if show_probs:
                        for i, p in enumerate(severity_probs):
                            st.write(f"{SEVERITY_NAMES[i]}: {p * 100:.2f}%")
                            st.progress(float(p))
                except Exception as e:
                    st.error(f"Severity model failed: {e}")

            if pred == 1 and ordinal_model is None:
                st.warning("Ordinal model not available.")

st.success("Done.")
