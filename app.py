# ========================= FINAL WORKING CODE (YOLO BOUNDING BOX + FIXED ORDINAL + GRADCAM) =========================

import os
import tempfile
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

# ================= YOLO IMPORT =================
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except Exception:
    HAS_ULTRALYTICS = False

# ================= CONFIG =================
st.set_page_config(page_title="GI Diagnosis System", layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

CLASS_NAMES = ["Normal", "Ulcerative Colitis", "Polyps", "Esophagitis"]
SEVERITY_NAMES = ["Remission", "Mild", "Moderate", "Severe"]

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATHS = {
    "classifier": os.path.join(MODEL_DIR, "best_reseff_fusion_masked.pth"),
    "ordinal": os.path.join(MODEL_DIR, "best_reseff_fusion_masked_ordinal.pth"),
    "polyp": os.path.join(MODEL_DIR, "best.pt"),
}

DRIVE_IDS = {
    "classifier": "1aZke_47izApUtev2-Jlr1j4ZC84DC1i1",
    "ordinal": "1Q74a7he0LnLfDJEN90YLpMhzJxI0wKa2",
    "polyp": "1xzGUJ1d9qDQKiodCzbWVOH07gNsffWnS",
}

# ================= MODEL =================
class ResEffFusion(nn.Module):
    def __init__(self, num_classes=4, eff_weight=0.75):
        super().__init__()
        self.eff = timm.create_model("efficientnet_b4", pretrained=False, features_only=True)
        self.res = timm.create_model("resnet50", pretrained=False, features_only=True)

        eff_dim = self.eff.feature_info[-1]["num_chs"]
        res_dim = self.res.feature_info[-1]["num_chs"]

        self.eff_proj = nn.Conv2d(eff_dim, 1024, 1)
        self.res_proj = nn.Conv2d(res_dim, 1024, 1)

        self.bn = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)

        self.eff_weight = eff_weight
        self.res_weight = 1 - eff_weight

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

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

# ================= DOWNLOAD =================
def download_if_missing(file_id, path):
    if os.path.exists(path):
        return True
    try:
        gdown.download(f"https://drive.google.com/uc?id={file_id}", path, quiet=False)
        return True
    except:
        return False

# ================= LOAD =================
@st.cache_resource
def load_classifier(path):
    model = ResEffFusion(4).to(DEVICE)
    sd = torch.load(path, map_location=DEVICE)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

@st.cache_resource
def load_ordinal(path):
    sd = torch.load(path, map_location=DEVICE)
    out_features = sd["classifier.weight"].shape[0]

    if out_features == 4:
        model = ResEffFusion(4).to(DEVICE)
    else:
        model = ResEffFusion(4).to(DEVICE)

    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, out_features

@st.cache_resource
def load_yolo(path):
    if not HAS_ULTRALYTICS:
        return None
    try:
        return YOLO(path)
    except:
        return None

# ================= PREDICT =================
def predict(model, img):
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return np.argmax(probs), probs

def predict_ordinal(model_info, img):
    model, out_features = model_info
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    return np.argmax(probs), probs

# ================= GRADCAM =================
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        model.bn.register_forward_hook(self.fwd)
        model.bn.register_backward_hook(self.bwd)

    def fwd(self, m, i, o):
        self.activations = o

    def bwd(self, m, gi, go):
        self.gradients = go[0]

    def generate(self, x, idx):
        self.model.zero_grad()
        out = self.model(x)
        out[:, idx].backward()

        w = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (w * self.activations).sum(dim=1).squeeze()

        cam = torch.relu(cam).detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

def overlay_cam(img, cam):
    img = cv2.resize(np.array(img), (IMG_SIZE, IMG_SIZE))
    heat = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heat, 0.4, 0)

# ================= YOLO (BOUNDING BOX) =================
def draw_boxes(img, boxes):
    draw = ImageDraw.Draw(img)
    w, h = img.size

    for box in boxes:
        x1, y1, x2, y2, conf = box

        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)
        draw.text((x1, y1-10), f"{conf:.2f}", fill="lime")

    return img

def run_yolo(model, img, conf):
    if model is None:
        return None

    results = model.predict(source=np.array(img), conf=conf, verbose=False)[0]

    boxes = []
    for b in results.boxes.data.tolist():
        x1, y1, x2, y2, confv, cls = b
        boxes.append((x1, y1, x2, y2, confv))

    return draw_boxes(img.copy(), boxes)

# ================= UI =================
st.title("GI Diagnosis System")

for k in MODEL_PATHS:
    download_if_missing(DRIVE_IDS[k], MODEL_PATHS[k])

clf = load_classifier(MODEL_PATHS["classifier"])
ord_model = load_ordinal(MODEL_PATHS["ordinal"])
yolo = load_yolo(MODEL_PATHS["polyp"])

uploaded = st.file_uploader("Upload Image", type=["jpg","png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img)

    pred, probs = predict(clf, img)
    st.write("Prediction:", CLASS_NAMES[pred])

    # GradCAM
    x = transform(img).unsqueeze(0).to(DEVICE)
    cam = GradCAM(clf).generate(x, pred)
    st.image(overlay_cam(img, cam), caption="GradCAM")

    # YOLO bounding box
    if pred == 2:
        det = run_yolo(yolo, img, 0.25)
        if det:
            st.image(det, caption="YOLO Bounding Boxes")

    # Severity
    if pred == 1:
        sev, _ = predict_ordinal(ord_model, img)
        st.write("Severity:", SEVERITY_NAMES[sev])
