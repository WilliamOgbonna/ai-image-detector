# Lightweight & Local AI Image Detector

You can upload an image and check whether it looks AI-generated, and which model likely created it.

It will predict two things:

- **Verdict**: real vs AI-generated
- **Source**: real, SD 2.1, SDXL, SD3, DALL·E 3, Midjourney, etc.

## Setup 🧰

```bash
python -m venv .venv
source .venv/bin/activate   # For Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run GUI 🏃‍♂️

```bash
python gui.py
```

1. Choose out of 3 detector modes (deep / manual / hybrid)
2. Upload an image
3. Click **Analyse**

Requires trained checkpoints in `checkpoints/`(currently some in repo):

- `mobilenetv3_detector.pth`: deep model
- `manual_detector.joblib`: manual model

## Training 🥊

Training gets the dataset from Hugging Face below on first run (connected independent of the repo):

**Dataset:** [Rajarshi-Roy-research/Defactify_Image_Dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset)

```bash
python train.py          # deep MobileNet model
python train_manual.py   # manual feature + logistic regression model
```

Checkpoints are saved to `checkpoints/`.


