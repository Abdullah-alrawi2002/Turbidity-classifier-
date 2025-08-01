from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps
import io
import numpy as np
import os

# --------- Config ----------
CLASSES = [
    "Ultra cloudy",
    "very cloudy",
    "cloudy",
    "lightly cloudy",
    "lightly clear",
    "clear",
]
TURBIDITY_RANGES = {
    "Ultra cloudy":   (3336.0, 3844.0),
    "very cloudy":    (1300.0, 2520.0),
    "cloudy":         (600.0, 1200.0),
    "lightly cloudy": (150.0, 450.0),
    "lightly clear":  (25.0, 90.0),
    "clear":          (1.47, 17.13),
}
MODEL_PATH = "best_turbidity_model.pth"

def gray_world(img: Image.Image) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    mean = arr.reshape(-1, 3).mean(0) + 1e-6
    arr *= mean.mean() / mean
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

preprocess = transforms.Compose([
    transforms.Lambda(lambda im: ImageOps.exif_transpose(im)),
    transforms.Lambda(gray_world),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def build_model(num_classes=len(CLASSES)):
    try:
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    except Exception:
        m = models.resnet34(pretrained=True)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_f, 512),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    return m

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model()
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} not found. Place the trained model in the backend folder.")
state = torch.load(MODEL_PATH, map_location=device)
if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
    state = state["state_dict"]
if any(k.startswith("module.") for k in state):
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
model.load_state_dict(state, strict=True)
model = model.to(device)
model.eval()

app = Flask(__name__)
CORS(app)

def predict_turbidity(image: Image.Image):
    x = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    pred_class = CLASSES[idx]
    confidence = float(probs[idx])
    ntu_min, ntu_max = TURBIDITY_RANGES[pred_class]
    per_class = {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}
    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "ntu_range": [ntu_min, ntu_max],
        "per_class_probs": per_class,
    }

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "No image part"}), 400
    file = request.files["image"]
    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": "Cannot process image", "detail": str(e)}), 400
    result = predict_turbidity(image)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
