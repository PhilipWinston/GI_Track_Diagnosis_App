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
try:
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except:
    HAS_ULTRALYTICS = False

st.set_page_config(page_title="GI Diagnosis Framework", layout="wide", initial_sidebar_state="expanded")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
DISPLAY_WIDTH = 420
CLASS_NAMES = ["Normal", "Ulcerative Colitis", "Polyps", "Esophagitis"]
CLASS_COLORS = ["#22c55e", "#eab308", "#3b82f6", "#a855f7"]
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

DRIVE_IDS = {
    "classifier": "1cvDPCfVBLWjCtx9mjz0KFWCwTwRs2atL",
    "ordinal": "1Hvng1F6upAUjfsZe_sLpTmmH8lia04TI",
    "polyp": "1xzGUJ1d9qDQKiodCzbWVOH07gNsffWnS",
}
MODEL_PATHS = {k: os.path.join(MODEL_DIR, f"best_{k}.pth" if k!="polyp" else "best.pt") for k in DRIVE_IDS}

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

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])

class ScoreCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        target_layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.activations = out.detach()

    def get_robust_mask(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (11,11), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((15,15), np.uint8)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h,w = gray.shape
        final = np.zeros((h,w), dtype=np.uint8)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(final, [largest], -1, 255, -1)
        else:
            cv2.circle(final, (w//2, h//2), int(min(w,h)*0.45), 255, -1)
        final = cv2.GaussianBlur(final, (25,25), 0)
        return final.astype(np.float32)/255.0

    def generate(self, x, class_idx, original_img=None, max_maps=6):
        self.model.eval()
        with torch.no_grad():
            _ = self.model(x)
        if self.activations is None: return None
        maps = torch.relu(self.activations[0])
        cam = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        for i in range(min(max_maps, maps.shape[0])):
            m = maps[i]
            mmin, mmax = m.min(), m.max()
            if (mmax-mmin).abs()<1e-6: continue
            m = (m-mmin)/(mmax-mmin+1e-8)
            m_up = cv2.resize(m.cpu().numpy(), (IMG_SIZE, IMG_SIZE))
            m_tensor = torch.from_numpy(m_up).float().to(DEVICE)
            masked = x * m_tensor.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                out = self.model(masked)
                score = float(torch.softmax(out,dim=1)[0,class_idx].item())
            cam += score * m_up
        if cam.max()<=0: return None
        cam /= (cam.max()+1e-8)
        if original_img is not None:
            orig = np.array(original_img.resize((IMG_SIZE,IMG_SIZE)))
            mask = self.get_robust_mask(orig)
            cam = cam * mask
            if cam.max()>0: cam /= (cam.max()+1e-8)
        cam = np.clip(cam, 0.55, 1.0)
        cam = cv2.GaussianBlur(cam, (9,9), 0)
        return cam
        
def download_if_missing(file_id: str, out_path: str, desc: str):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1024: return True
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        with st.spinner(f"Downloading {desc}..."):
            gdown.download(url, out_path, quiet=False)
        return os.path.exists(out_path) and os.path.getsize(out_path) > 1024
    except:
        return False

@st.cache_resource
def load_classification_model(path: str):
    m = EFFResNetViT(num_classes=4).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m

@st.cache_resource
def load_ordinal_model(path: str):
    m = EFFResNetViTOrdinal(num_classes=4).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m

@st.cache_resource
def load_yolo_model(path: str):
    if not HAS_ULTRALYTICS: return (None, None)
    try:
        return ("ultralytics", YOLO(path))
    except:
        return (None, None)

def predict_class(model, pil_img):
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(probs.argmax())
        conf = float(probs[pred])
    return pred, conf, x

def predict_ordinal(ordinal_model, pil_img):
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = ordinal_model(x)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        label = int((probs > 0.5).sum())
    return label, probs

def run_yolo(yolo_loader, src_image, conf=0.25):
    if yolo_loader[0] != "ultralytics": return []
    img_bgr = cv2.cvtColor(np.array(src_image.resize((IMG_SIZE*2, IMG_SIZE*2))), cv2.COLOR_RGB2BGR)
    results = yolo_loader[1].predict(source=img_bgr, conf=conf, verbose=False)
    r = results[0]
    boxes = []
    for b in r.boxes.data.tolist():
        x1,y1,x2,y2,confv,cls = b
        boxes.append({"xyxy": np.array([x1,y1,x2,y2]), "conf": confv})
    return boxes

def overlay_boxes(image, boxes):
    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    sx = w / (IMG_SIZE*2)
    sy = h / (IMG_SIZE*2)
    for b in boxes:
        x1,y1,x2,y2 = (int(v*s) for v,s in zip(b["xyxy"], [sx,sy,sx,sy]))
        cv2.rectangle(arr, (x1,y1), (x2,y2), (0,255,0), 3)
        cv2.putText(arr, f"{b['conf']:.2f}", (x1, max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return Image.fromarray(arr)

st.markdown("""
<style>
    .main {background: linear-gradient(180deg, #0f172a 0%, #1e2937 100%);}
    h1 {color: #60a5fa; text-align: center; font-family: 'Helvetica Neue', sans-serif; font-weight: 700; margin-bottom: 0;}
    .subtitle {text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;}
    .card {background: #1e2937; padding: 24px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); border: 1px solid #334155;}
    .badge {padding: 12px 28px; border-radius: 50px; font-size: 1.6rem; font-weight: 700; display: inline-block; margin: 12px 0;}
    .img-container {border: 3px solid #334155; border-radius: 16px; overflow: hidden;}
    .section-header {color: #60a5fa; border-bottom: 2px solid #334155; padding-bottom: 8px;}
    .severity {font-size: 1.3rem; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

st.title("Deep Learning Framework for Explainable Diagnosis of GI Tract Disorders")
st.markdown('<p class="subtitle">EFFResNetViT • Score-CAM • Ordinal Severity • YOLO Polyp Detection</p>', unsafe_allow_html=True)

st.sidebar.header("Configuration")
show_scorecam = st.sidebar.checkbox("Enable Score-CAM", value=True)
conf_threshold = st.sidebar.slider("YOLO Confidence", 0.05, 0.95, 0.25, 0.01)
st.sidebar.markdown("### Model Status")

dl_ok = {}
for k in DRIVE_IDS:
    ok = download_if_missing(DRIVE_IDS[k], MODEL_PATHS[k], k)
    dl_ok[k] = ok
    if ok:
        st.sidebar.success(f"✅ {k.capitalize()}")
    else:
        st.sidebar.error(f"❌ {k.capitalize()}")

classification_model = load_classification_model(MODEL_PATHS["classifier"]) if dl_ok.get("classifier") and os.path.exists(MODEL_PATHS["classifier"]) else None
ordinal_model = load_ordinal_model(MODEL_PATHS["ordinal"]) if dl_ok.get("ordinal") and os.path.exists(MODEL_PATHS["ordinal"]) else None
yolo_loader = load_yolo_model(MODEL_PATHS["polyp"]) if dl_ok.get("polyp") and os.path.exists(MODEL_PATHS["polyp"]) and HAS_ULTRALYTICS else (None, None)

uploaded = st.file_uploader("Upload colonoscopy images (JPG/PNG)", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded:
    progress = st.progress(0)
    for idx, f in enumerate(uploaded):
        img = Image.open(f).convert("RGB")
        col1, col2 = st.columns([1, 1.1])
        
        with col1:
            st.markdown('<div class="img-container">', unsafe_allow_html=True)
            st.image(img, caption=f"Image {idx+1}", width=DISPLAY_WIDTH)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if not classification_model:
                st.error("Classifier not loaded")
                st.markdown('</div>', unsafe_allow_html=True)
                continue
            
            pred, conf, x = predict_class(classification_model, img)
            color = CLASS_COLORS[pred]
            name = CLASS_NAMES[pred]
            
            st.markdown(f'<div class="badge" style="background:{color};color:white;">{name}</div>', unsafe_allow_html=True)
            st.progress(conf, text=f"Confidence: {conf*100:.1f}%")
            
            # Polyp detection
            if pred == 2:
                if yolo_loader[0] == "ultralytics":
                    boxes = run_yolo(yolo_loader, img, conf_threshold)
                    if boxes:
                        over = overlay_boxes(img, boxes)
                        st.image(over, caption="YOLO Polyp Detections", width=DISPLAY_WIDTH)
                        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                        over.save(tmp.name)
                        st.download_button("📥 Download Detections", open(tmp.name,"rb"), f"detections_{idx+1}.png", key=f"download_detections_{idx}")
                    else:
                        st.info("No polyps detected")
                else:
                    st.info("YOLO unavailable")
            
            # UC severity
            if pred == 1 and ordinal_model:
                label, _ = predict_ordinal(ordinal_model, img)
                severity_text = ["Remission", "Mild", "Moderate", "Severe"][label]
                severity_color = ["#22c55e", "#eab308", "#f97316", "#ef4444"][label]
                st.markdown(f'<div class="severity" style="color:{severity_color};">Severity: {severity_text} ({label}/3)</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Score-CAM
            if show_scorecam and classification_model:
                try:
                    sc = ScoreCAM(classification_model, classification_model.fusion)
                    cam = sc.generate(x, pred, original_img=img)
                    if cam is not None:
                        heat = cv2.applyColorMap((cam*255).astype("uint8"), cv2.COLORMAP_TURBO)
                        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
                        img_res = np.array(img.resize((IMG_SIZE,IMG_SIZE))).astype("uint8")
                        overlay = np.uint8(0.65 * img_res + 0.35 * heat)
            
                        st.markdown('<div class="card"><p class="section-header">Score-CAM Explainability (Notebook Quality)</p>', unsafe_allow_html=True)
                        st.image(heat, caption="Score-CAM Heatmap", width=DISPLAY_WIDTH)
                        st.image(overlay, caption="Overlay", width=DISPLAY_WIDTH)
                        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                        Image.fromarray(overlay).save(tmp.name)
                        st.download_button("📥 Download Overlay", open(tmp.name,"rb"), f"scorecam_{idx+1}.png", key=f"sc_{idx}")
                        st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Score-CAM: {str(e)[:80]}")
        
        progress.progress((idx+1)/len(uploaded))
    progress.empty()

else:
    st.info("👆 Upload images to begin analysis")

st.caption("© 2026 Deep Learning Framework for Explainable Diagnosis of GI Tract Disorders • Models from provided Google Drive")















