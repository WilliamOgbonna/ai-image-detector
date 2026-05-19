from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from manual_features import extract_manual_features
from train import CHECKPOINT_DIR, DEEP_CHECKPOINT_PATH, MobileNetV3DualHead, build_transforms

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp",
                 ".tiff", ".tif", ".gif")
MANUAL_CHECKPOINT_PATH = CHECKPOINT_DIR / "manual_detector.joblib"


#For detetcor's verdict and source for user
@dataclass
class PredictionResult:
    verdict_label: str
    verdict_confidence: float
    source_label: str
    source_confidence: float
    mode: str

#The Detection Options from checkpoints 
class Detector:
    
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dp_checked = False
        self._ml_checked = False
        self.dp_model = None
        self.dp_verdict_lbls = []
        self.dp_source_lbls = []
        self.dp_transform = build_transforms(train=False)
        self.ml_scaler = None
        self.ml_verdict_model = None
        self.ml_source_model = None
        self.ml_verdict_lbls = []
        self.ml_source_lbls = []
   #Checks which model can be used 
    def available_modes(self) -> Dict[str, bool]:
        deep = DEEP_CHECKPOINT_PATH.exists()
        manual = MANUAL_CHECKPOINT_PATH.exists()
        return {
            "deep": deep,
            "manual": manual,
            "hybrid": deep and manual,
        }
   #Checks model can be used 
    def _check_ml(self) -> None:
        if self._ml_checked:
            return
        if not MANUAL_CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Manual checkpoint missing: {MANUAL_CHECKPOINT_PATH}")

        payload = joblib.load(MANUAL_CHECKPOINT_PATH)
        self.ml_scaler = payload["scaler"]
        self.ml_verdict_model = payload["verdict_model"]
        self.ml_source_model = payload["source_model"]
        self.ml_verdict_lbls = payload["verdict_labels"]
        self.ml_source_lbls = payload["source_labels"]
        self._ml_checked = True
    #checks manual model can be used 
    def _check_dp(self) -> None:
        if self._dp_checked:
            return

        if not DEEP_CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"Error, Unfortunately, The deep checkpoint isnt found: {DEEP_CHECKPOINT_PATH}")
        payload = torch.load(DEEP_CHECKPOINT_PATH, map_location=self.device)
        source_labels = payload["source_labels"]
        model = MobileNetV3DualHead(verdict_types=2, source_types=len(source_labels))
        model.load_state_dict(payload["model_state_dict"])
        model.to(self.device).eval()

        self.dp_model = model
        self.dp_verdict_lbls = payload["verdict_labels"]
        self.dp_source_lbls = source_labels
        self._dp_checked = True
    #predicts user image using deep 
    def _predict_dp(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        self._check_dp()
        x = self.dp_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            verdict_logits, source_logits = self.dp_model(x)
            verdict_probs = F.softmax(verdict_logits, dim=1).cpu().numpy()[0]
            source_probs = F.softmax(source_logits, dim=1).cpu().numpy()[0]
        return verdict_probs, source_probs
    #preicts users iage using manual model
    def _predict_ml(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
        self._check_ml()
        ml_feats = extract_manual_features(image).reshape(1, -1)
        ml_feats_scaled = self.ml_scaler.transform(ml_feats)
        ml_vp = self.ml_verdict_model.predict_proba(ml_feats_scaled)[0]
        ml_sp = self.ml_source_model.predict_proba(ml_feats_scaled)[0]
        return ml_vp, ml_sp
    #keeps the detection resulst for user
    def _save_results(
        self,
        mode: str,
        vp: np.ndarray,
        sp: np.ndarray,
        v_lbls: list[str],
        s_lbls: list[str],
    ) -> PredictionResult:
        v_idx = int(np.argmax(vp))
        s_idx = int(np.argmax(sp))
        return PredictionResult(
            verdict_label=v_lbls[v_idx],
            verdict_confidence=float(vp[v_idx]) * 100.0,
            source_label=s_lbls[s_idx],
            source_confidence=float(sp[s_idx]) * 100.0,
            mode=mode,
        )
    #Hybrid Detection Option 
    def predict(self, image: Image.Image, mode: str = "hybrid") -> PredictionResult:
        mode = mode.lower().strip()
        availability = self.available_modes()
        #Ensure user chooses one fo the detector options
        if mode not in availability:
            raise ValueError(f" This is invalid '{mode}'. Please choose between deep,manual,hybrid.")
        #block a model that isnt currently available
        if not availability[mode]:
            raise ValueError(f"The option '{mode}'  currently is unavailable. You will need to Kindly train or provide required checkpoints.")

        if mode == "deep":
            vp, sp = self._predict_dp(image)
            return self._save_results(mode, vp, sp, self.dp_verdict_lbls, self.dp_source_lbls)

        if mode == "manual":
            vp, sp = self._predict_ml(image)
            return self._save_results(mode, vp, sp, self.ml_verdict_lbls, self.ml_source_lbls)

        ml_vp, ml_sp = self._predict_ml(image)
        dp_vp, dp_sp = self._predict_dp(image)
        vp = 0.6 * dp_vp + 0.4 * ml_vp
        sp = 0.6 * dp_sp + 0.4 * ml_sp
        return self._save_results(mode, vp, sp, self.dp_verdict_lbls, self.dp_source_lbls)

    #Ensure the right image type 
def _is_supported_image(path: str) -> bool:
    return path.lower().endswith(SUPPORTED_EXTENSIONS)




def main() -> None:
    parser = argparse.ArgumentParser(description="Inference Detector")
    parser.add_argument("--image", required=True, help="Path to user'sinput image.")
    parser.add_argument("--mode", default="hybrid", choices=["deep", "manual", "hybrid"])
    args = parser.parse_args()

      #For Wrong File Type
    if not _is_supported_image(args.image):
        raise ValueError(f"Unsupported file extension for '{args.image}'")

    image = Image.open(args.image).convert("RGB")
    result = Detector().predict(image, mode=args.mode)

    print(f"Mode: {result.mode}")
    print(f"Verdict: {result.verdict_label} ({result.verdict_confidence:.2f}%)")
    print(f"Source : {result.source_label} ({result.source_confidence:.2f}%)")
 
if __name__ == "__main__":
    main()
