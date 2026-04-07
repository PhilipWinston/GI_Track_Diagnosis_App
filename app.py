# ========================= FINAL FIXED WORKING CODE =========================
# ✅ YOLOv11 custom model bounding box FIXED
# ✅ GradCAM FIXED
# ✅ Ordinal FIXED

import os
import cv2
import gdown
import numpy as np
import streamlit as st
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# ================= YOLO =================
from ultralytics import YOLO   # assume installed properly

# ================= CONFIG =================
st.set_page_config(page_title="GI Diagnosis", layout="wide")

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
    def __init__(self, num_classes=4):
        super().__init__()
        self.eff = timm.create_model("efficientnet_b4", pretrained=False, features_only=True)
        self.res = timm.create_model("resnet50", pretrained=False, features_only=True)

        eff_dim = self.eff.feature_info[-1]["num_chs"]
        res_dim = self.res.feature_info[-1]["num_chs"]

        self.eff_proj = nn.Conv2d(eff_dim, 1024, 1)
        self.res_proj = nn.Conv2d(res_dim, 1024, 1)

        self.bn = nn.BatchNorm2d(1024)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        eff = self.eff(x)[-1]
        res = self.res(x)[-1]

        if eff.shape[2:] != res.shape[2:]:
            res = nn.functional.interpolate(res, size=eff.shape[2:], mode="bilinear")

        fused = self.relu(self.bn(self.eff_proj(eff) + self.res_proj(res)))
        return self.classifier(self.pool(fused).flatten(1))

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
    model.load_state_dict(torch.load(path, map_location=DEVICE), strict=False)
    model.eval()
    return model

@st.cache_resource
def load_ordinal(path):
    model = ResEffFusion(4).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE), strict=False)
    model.eval()
    return model

@st.cache_resource
def load_yolo(path):
    return YOLO(path)

# ================= PREDICT =================
def predict(model, img):
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    return np.argmax(probs), probs

# ================= GRADCAM =================
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.grad = None
        self.act = None

        model.bn.register_forward_hook(lambda m,i,o: setattr(self,"act",o))
        model.bn.register_backward_hook(lambda m,gi,go: setattr(self,"grad",go[0]))

    def generate(self, x, idx):
        self.model.zero_grad()
        out = self.model(x)
        out[:, idx].backward()

        w = self.grad.mean(dim=(2,3), keepdim=True)
        cam = (w * self.act).sum(dim=1).squeeze()

        cam = torch.relu(cam).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

def overlay_cam(img, cam):
    img = cv2.resize(np.array(img), (IMG_SIZE, IMG_SIZE))
    heat = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heat, 0.4, 0)

# ================= YOLO FIX (IMPORTANT) =================
def run_yolo(model, img, conf=0.25):
    results = model(img, conf=conf)[0]

    img_np = np.array(img).copy()

    if results.boxes is None or len(results.boxes) == 0:
        return img, 0

    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(img_np, (x1,y1), (x2,y2), (0,255,0), 3)
        cv2.putText(img_np, f"{score:.2f}", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    return Image.fromarray(img_np), len(boxes)

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

    pred, _ = predict(clf, img)
    st.write("Prediction:", CLASS_NAMES[pred])

    # GradCAM
    x = transform(img).unsqueeze(0).to(DEVICE)
    cam = GradCAM(clf).generate(x, pred)
    st.image(overlay_cam(img, cam), caption="GradCAM")

    # 🔥 ALWAYS RUN YOLO (IMPORTANT FIX)
    st.subheader("YOLOv11 Detection")

    det_img, count = run_yolo(yolo, img)

    if count > 0:
        st.success(f"{count} object(s) detected")
    else:
        st.warning("No detections")

    st.image(det_img, caption="Bounding Boxes")

    # Severity
    if pred == 1:
        sev, _ = predict(ord_model, img)
        st.write("Severity:", SEVERITY_NAMES[sev])
