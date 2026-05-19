from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from manual_features import extract_manual_features
from train import load_hf_split #loads dataste's split to use manual training 

VERDICTS = ["real", "ai-generated"]
SOURCES = ["real", "sd21", "sdxl", "sd3", "dalle3", "midjourney"]
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
MANUAL_CHECKPOINT_PATH = CHECKPOINT_DIR / "manual_detector.joblib"

#Builds matrix for the manual features, verdict and source types 
def build_feature_matrix(
    hf_ds, 
    image_key: str, 
    label1_key: str, 
    label2_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = []
    y_verdict = []
    y_source = []
    for row in tqdm(hf_ds, desc="Extracting manual features"):
        image_obj = row[image_key]
        xs.append(extract_manual_features(image_obj.convert("RGB")))
        y_verdict.append(int(row[label1_key]))
        y_source.append(int(row[label2_key]))
    return ( np.asarray(xs, dtype=np.float32), 
    np.asarray(y_verdict), 
    np.asarray(y_source) )

#Metrcis to assess the manual detction option 
def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print(f"\n== {name} ==")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1       : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train manual detector.")
    parser.add_argument("--dataset", default="Rajarshi-Roy-research/Defactify_Image_Dataset")
    parser.add_argument("--max-train-samples", type=int, default=30000)
    parser.add_argument("--max-test-samples", type=int, default=8000)
    args = parser.parse_args()

    train_hf, img_k, l1_k, l2_k = load_hf_split(
        args.dataset, "train", args.max_train_samples
        )

    test_hf, _, _, _ = load_hf_split(
        args.dataset, "test", args.max_test_samples
        )

    x_train, yv_train, ys_train = build_feature_matrix(
        train_hf, img_k, l1_k, l2_k
        )
    x_test, yv_test, ys_test = build_feature_matrix(
        test_hf, img_k, l1_k, l2_k
        )
    # scaling features 
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    #training for verdict and source decisions 
    verdict_model = LogisticRegression(max_iter=300, n_jobs=-1)
    source_model = LogisticRegression(max_iter=300, n_jobs=-1)

    verdict_model.fit(x_train_s, yv_train)
    source_model.fit(x_train_s, ys_train)

    verdict_pred = verdict_model.predict(x_test_s)
    source_pred = source_model.predict(x_test_s)

    evaluate("Verdict ", yv_test, verdict_pred)
    evaluate("Source ", ys_test, source_pred)

    num_source_classes = int(np.max(ys_train)) + 1
    source_labels = SOURCES[:num_source_classes]
    if len(source_labels) < num_source_classes:
        source_labels.extend([f"source_{i}" for i in range(len(source_labels), num_source_classes)])

    payload = {
        "scaler": scaler,
        "verdict_model": verdict_model,
        "source_model": source_model,
        "verdict_labels": VERDICTS,
        "source_labels": source_labels,
    }
    joblib.dump(payload, MANUAL_CHECKPOINT_PATH)
    print(f"\nSaved manual model checkpoint: {MANUAL_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
