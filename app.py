import os
import tempfile
from pathlib import Path

import gdown
import numpy as np
import streamlit as st
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False


# ============================================================================
# APP CONFIG
# ============================================================================
st.set_page_config(
    page_title="GI Tract Diagnosis System",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


def download_if_missing(file_id: str, out_path: str) -> bool:
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024:
        return True
    try:
        gdown.download(f"https://drive.google.com/uc?id={file_id}", out_path, quiet=False)
        return os.path.exists(out_path) and os.path.getsize(out_path) > 1024
    except Exception:
        return False


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
    try:
        from ultralytics import YOLO
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"YOLO load error: {e}")
        return None


# ============================================================================
# PREDICTION
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


def run_yolo(yolo_model, image: Image.Image, conf: float):
    if yolo_model is None:
        return []

    try:
        img_np = np.array(image.convert("RGB"))

        results = yolo_model.predict(
            source=img_np,
            conf=conf,
            imgsz=640,
            verbose=False,
        )

        boxes = []
        if len(results) == 0:
            return boxes

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return boxes

        for b in r.boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            conf_val = float(b.conf[0].cpu().numpy())
            boxes.append({"xyxy": xyxy, "conf": conf_val})

        return boxes

    except Exception as e:
        st.error(f"YOLO inference error: {e}")
        return []


def overlay_boxes(image: Image.Image, boxes: list):
    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box["xyxy"]
        conf = box["conf"]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        label = f"Polyp {i + 1}: {conf:.2f}"

        try:
            tb = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill="black")
        except Exception:
            pass

        draw.text((x1, max(0, y1 - 12)), label, fill="white", font=font)

    return img


# ============================================================================
# UI
# ============================================================================
st.title("GI Tract Diagnosis System")
st.caption("Simple working version for classification, UC severity, and polyp detection")

st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("YOLO confidence", 0.05, 0.95, 0.25, 0.05)
show_details = st.sidebar.checkbox("Show probabilities", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Status")

dl_ok = {}
for name, file_id in DRIVE_IDS.items():
    ok = download_if_missing(file_id, MODEL_PATHS[name])
    dl_ok[name] = ok
    st.sidebar.write(f"{'✅' if ok else '❌'} {name}")

st.sidebar.markdown("---")
st.sidebar.write(f"Device: {DEVICE.type.upper()}")


# ============================================================================
# LOAD MODELS
# ============================================================================
classifier_model = None
ordinal_model = None
yolo_model = None

try:
    if dl_ok.get("classifier") and os.path.exists(MODEL_PATHS["classifier"]):
        classifier_model, _ = load_torch_model(MODEL_PATHS["classifier"], default_outputs=4)
except Exception as e:
    st.sidebar.error(f"Classifier load error: {e}")
    classifier_model = None

try:
    if dl_ok.get("ordinal") and os.path.exists(MODEL_PATHS["ordinal"]):
        ordinal_model, _ = load_torch_model(MODEL_PATHS["ordinal"], default_outputs=3)
except Exception as e:
    st.sidebar.error(f"Ordinal load error: {e}")
    ordinal_model = None

try:
    if dl_ok.get("polyp") and os.path.exists(MODEL_PATHS["polyp"]):
        yolo_model = load_yolo_model(MODEL_PATHS["polyp"])
except Exception as e:
    st.sidebar.error(f"YOLO load error: {e}")
    yolo_model = None


# ============================================================================
# UPLOADER
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
# PROCESS
# ============================================================================
for idx, f in enumerate(uploaded_files):
    image = Image.open(f).convert("RGB")

    st.divider()
    st.subheader(f"Image {idx + 1}")

    left, right = st.columns([1, 1.15])

    with left:
        st.image(image, use_container_width=True)

    with right:
        if classifier_model is None:
            st.error("Classification model not loaded.")
        else:
            pred, conf, probs = predict_classification(classifier_model, image)
            st.write(f"**Prediction:** {CLASS_NAMES[pred]}")
            st.write(f"**Confidence:** {conf * 100:.2f}%")

            if show_details:
                for i, p in enumerate(probs):
                    st.write(f"{CLASS_NAMES[i]}: {p * 100:.2f}%")
                    st.progress(float(p))

            if pred == 2:
                st.write("**Polyp detection**")
                if yolo_model is None:
                    st.warning("YOLO model not available.")
                else:
                    boxes = run_yolo(yolo_model, image, conf_threshold)
                    if boxes:
                        st.success(f"{len(boxes)} polyp(s) detected")
                        det_img = overlay_boxes(image, boxes)
                        st.image(det_img, caption="YOLO detection", use_container_width=True)

                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                        det_img.save(tmp.name)
                        with open(tmp.name, "rb") as fp:
                            st.download_button(
                                "Download detection image",
                                data=fp,
                                file_name=f"polyp_detection_{idx + 1}.png",
                                mime="image/png",
                                key=f"download_{idx}",
                            )
                    else:
                        st.info("No polyp detected.")

            if pred == 1:
                st.write("**UC severity**")
                if ordinal_model is None:
                    st.warning("Ordinal model not loaded.")
                else:
                    try:
                        severity_label, severity_probs = predict_ordinal(ordinal_model, image)
                        st.write(f"**Severity:** {SEVERITY_NAMES[severity_label]}")
                        st.write(f"**Mayo grade:** {severity_label}/3")

                        if show_details:
                            for i, p in enumerate(severity_probs):
                                st.write(f"{SEVERITY_NAMES[i]}: {p * 100:.2f}%")
                                st.progress(float(p))
                    except Exception as e:
                        st.error(f"Severity model failed: {e}")

st.success("Done.")
