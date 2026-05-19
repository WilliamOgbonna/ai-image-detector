from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

#Normlizing histogram 
def _normalize_hist(hist: np.ndarray) -> np.ndarray:
    hist = hist.astype(np.float32)
    denom = hist.sum()
    if denom > 0:
        hist /= denom
    return hist

#Manual Feature 1 - Colour histogram with 96 dimensions 
def color_histogram_96(
    image: Image.Image, bins_per_channel: int = 32
    ) -> np.ndarray:
    arr = np.array(image.convert("RGB"))
    features = []
    for channel in range(3):
        hist, _ = np.histogram(arr[:, :, channel], bins=bins_per_channel, range=(0, 256))
        features.append(_normalize_hist(hist))
    return np.concatenate(features, axis=0)

#Manual Feature 2 - Local Binary Pattern histogram with 10 dimensions 
def lbp_histogram_10(image: Image.Image) -> np.ndarray:
    gray = np.array(image.convert("L"))
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    return _normalize_hist(hist)

#Manual Feature 3 - GRay Level Co-occurence Matrix with 3 
def glcm_features_3(image: Image.Image) -> np.ndarray:
    gray = np.array(image.convert("L"))
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    return np.array([contrast, homogeneity, energy], dtype=np.float32)

#Extracting manaul features into a feature vector 
def extract_manual_features(image: Image.Image) -> np.ndarray:
    color = color_histogram_96(image)
    lbp = lbp_histogram_10(image)
    glcm = glcm_features_3(image)
    return np.concatenate([color, lbp, glcm], axis=0).astype(np.float32)
