"""
Model loading and inference logic for the Wafer Defect Classifier.
Loaded once at startup and reused across requests.
"""
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
from pathlib import Path
from scipy.ndimage import zoom
from typing import List, Dict, Tuple

MODELS_DIR = Path(__file__).parent.parent / "models"


# ── CNN Architecture (must match training) ────────────────────────────

class WaferCNNEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, n_classes: int = 8):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.skip1 = nn.Conv2d(32, 64, 1, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.skip2 = nn.Conv2d(64, 128, 1, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x, return_embedding: bool = False):
        x1 = self.conv1(x)
        x2 = self.conv2(x1) + self.skip1(x1)
        x3 = self.conv3(x2) + self.skip2(x2)
        emb = self.embedding(self.gap(x3))
        if return_embedding:
            return emb
        return self.classifier(emb)


# ── Preprocessing ─────────────────────────────────────────────────────

def preprocess_wafer_map(wm: np.ndarray,
                          target_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """Resize and normalize a wafer map to model input format."""
    h, w = wm.shape
    resized = zoom(wm.astype(float),
                   (target_size[0] / h, target_size[1] / w), order=1)
    return np.clip(resized, 0, 1).astype(np.float32)


def map_to_tensor(wm: np.ndarray) -> torch.Tensor:
    """Convert preprocessed wafer map to (1, 3, 64, 64) tensor."""
    tensor = torch.from_numpy(wm).unsqueeze(0)  # (1, 64, 64)
    tensor = tensor.repeat(3, 1, 1).unsqueeze(0)  # (1, 3, 64, 64)
    # Normalize to [-1, 1]
    return (tensor - 0.5) / 0.5


def extract_domain_features(wm: np.ndarray) -> np.ndarray:
    """Extract 14 domain-engineered spatial features from a wafer map."""
    size = wm.shape[0]
    cx, cy = size // 2, size // 2
    radius = size // 2

    valid = np.array([
        [np.sqrt((i - cx) ** 2 + (j - cy) ** 2) <= radius for j in range(size)]
        for i in range(size)
    ])
    defective = wm > 0.4
    n_valid  = valid.sum()
    n_defect = (defective & valid).sum()
    density  = n_defect / n_valid if n_valid > 0 else 0

    zones = []
    for r0, r1 in [(0, 0.33), (0.33, 0.67), (0.67, 1.0)]:
        zone = np.array([
            [r0 * radius < np.sqrt((i-cx)**2+(j-cy)**2) <= r1 * radius
             for j in range(size)] for i in range(size)
        ])
        zv = (zone & valid).sum()
        zd = (zone & valid & defective).sum()
        zones.append(zd / zv if zv > 0 else 0)

    quads = []
    for qi, qj in [
        (slice(None, cx), slice(None, cy)), (slice(None, cx), slice(cy, None)),
        (slice(cx, None), slice(None, cy)), (slice(cx, None), slice(cy, None))
    ]:
        qv = valid[qi, qj].sum()
        qd = (defective & valid)[qi, qj].sum()
        quads.append(qd / qv if qv > 0 else 0)

    row_vars = np.var(wm * valid, axis=1)
    col_vars = np.var(wm * valid, axis=0)
    def_pos  = np.argwhere(defective & valid)
    compactness = 1 / (1 + np.mean(np.std(def_pos, axis=0))) if len(def_pos) > 1 else 0

    return np.array([
        density, *zones,
        np.std(quads), np.max(quads),
        zones[0] - zones[2],
        max(row_vars.max(), col_vars.max()),
        zones[1] - (zones[0] + zones[2]) / 2,
        compactness, *quads
    ])


# ── Model Registry ────────────────────────────────────────────────────

class WaferDefectClassifier:
    """
    Production inference wrapper for the wafer defect classification pipeline.
    Loads CNN + ensemble once at startup; thread-safe for concurrent requests.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_model  = None
        self.ensemble   = None
        self.scaler     = None
        self.le         = None
        self.metadata   = None
        self._loaded    = False

    def load(self):
        """Load all model artifacts from disk."""
        print(f"Loading models from {MODELS_DIR} on {self.device}...")

        # CNN encoder
        self.cnn_model = WaferCNNEncoder().to(self.device)
        self.cnn_model.load_state_dict(
            torch.load(MODELS_DIR / "cnn_encoder_best.pt", map_location=self.device)
        )
        self.cnn_model.eval()

        # Ensemble + scaler + encoder
        self.ensemble = joblib.load(MODELS_DIR / "ensemble_classifier.pkl")
        self.scaler   = joblib.load(MODELS_DIR / "feature_scaler.pkl")
        self.le       = joblib.load(MODELS_DIR / "label_encoder.pkl")

        # Metadata
        with open(MODELS_DIR / "metadata.json") as f:
            self.metadata = json.load(f)

        self._loaded = True
        print(f"Models loaded. Accuracy={self.metadata['model_accuracy']:.3f}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, wafer_map: List[List[float]]) -> Dict:
        """
        Run full inference pipeline on a single wafer map.

        Args:
            wafer_map: 2D list of floats (raw wafer map)

        Returns:
            dict with defect_class, confidence, top3, yield_estimate,
                  root_cause, priority, defect_density
        """
        if not self._loaded:
            raise RuntimeError("Models not loaded. Call load() first.")

        wm = np.array(wafer_map, dtype=np.float32)

        # Preprocess
        wm_proc = preprocess_wafer_map(wm)

        # CNN embedding
        tensor = map_to_tensor(wm_proc).to(self.device)
        with torch.no_grad():
            cnn_emb = self.cnn_model(tensor, return_embedding=True)
            cnn_emb = cnn_emb.cpu().numpy()  # (1, 128)

        # Domain features
        domain_feats = extract_domain_features(wm_proc).reshape(1, -1)  # (1, 14)

        # Combine + scale
        X = np.hstack([cnn_emb, domain_feats])  # (1, 142)
        X_scaled = self.scaler.transform(X)

        # Ensemble prediction
        proba = self.ensemble.predict_proba(X_scaled)[0]
        pred_idx  = proba.argmax()
        pred_class = self.le.classes_[pred_idx]
        confidence = float(proba[pred_idx])

        # Top-3
        top3_idx = proba.argsort()[::-1][:3]
        top3 = [
            {"class": self.le.classes_[i], "confidence": float(proba[i])}
            for i in top3_idx
        ]

        return {
            "defect_class"   : pred_class,
            "confidence"     : confidence,
            "top3"           : top3,
            "defect_density" : float(domain_feats[0, 0]),
            "yield_estimate" : float(max(0, 1 - domain_feats[0, 0])),
            "root_cause"     : self.metadata["root_causes"][pred_class],
            "priority"       : self.metadata["priority"][pred_class],
        }


# Singleton — loaded once at app startup
classifier = WaferDefectClassifier()
