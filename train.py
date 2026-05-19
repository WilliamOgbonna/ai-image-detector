from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision import transforms
from tqdm import tqdm

#Declaring The Classification Values & Paths 
IMAGE_SIZE = 224
VERDICTS = ["real", "AI-generated"]
SOURCES = ["real", "sd21", "sdxl", "sd3", "dalle3", "midjourney"]
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DEEP_CHECKPOINT_PATH = CHECKPOINT_DIR / "mobilenetv3_detector.pth"


#Structure for storing data from Dataset 
def _schema_keys(sample: dict) -> tuple[str, str, str]:
    if "Image" in sample:
        img_k = "Image"
    elif "image" in sample:
        img_k = "image"
    else:
        raise KeyError(f"There is no image column in: {list(sample.keys())}")
    l1 = None
    for k in ("Label_A", "label_1", "Label_1"):
        if k in sample:
            l1 = k
            break
    l2 = None
    for k in ("Label_B", "label_2", "Label_2"):
        if k in sample:
            l2 = k
            break
    if l1 is None or l2 is None:
        raise KeyError(f"Unexpected dataset error: {list(sample.keys())}")
    return img_k, l1, l2


def _pil_rgb(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.fromarray(img).convert("RGB")

#Loads dataset split 
def load_hf_split(dataset_name: str, split: str, max_samples: int | None = None):
    #Split dataset into train, validation, test section
    split_map = {"train": "train", "validation": "validation", "valid": "validation", "val": "validation", "test": "test"}
    selected_split = split_map[split.lower().strip()]
    ds = load_dataset(dataset_name, split=selected_split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    sample = ds[0]
    img_k, l1_k, l2_k = _schema_keys(sample)
    #returns the dataset, image, verdict and source data
    return ds, img_k, l1_k, l2_k

#For transforming and normlaizing the image groups 
def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

#Deep learning detection (MobileNetV3) for AI and source classification 
#images features are extracted and then seperated into heads 
class MobileNetV3DualHead(nn.Module):
    
    def __init__(self, verdict_types: int = 2, source_types: int = 6):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_features = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.dropout = nn.Dropout(p=0.2)
        self.verdict_hd = nn.Linear(in_features, verdict_types) #this head is used for the veerdict of AI or real images
        self.source_hd = nn.Linear(in_features, source_types) # this head is used for the model sources 

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        features = self.dropout(features)
        verdict_logits = self.verdict_hd(features)
        source_logits = self.source_hd(features)
        return verdict_logits, source_logits


#Dataset class for loading images from dataset in a light way 
class DefactifyHFDataset(Dataset):
    
   #define image data parameters 
    def __init__(
        self,
        hf_ds,
        transform: transforms.Compose,
        image_key: str,
        label1_key: str,
        label2_key: str,
    ):
    #transforms data
        self.hf_ds = hf_ds
        self.transform = transform
        self.image_key = image_key
        self.label1_key = label1_key
        self.label2_key = label2_key

    def __len__(self) -> int:
        return len(self.hf_ds)

    def __getitem__(self, idx: int):
        row = self.hf_ds[idx]
        img = _pil_rgb(row[self.image_key])
        y1 = int(row[self.label1_key])
        y2 = int(row[self.label2_key])
        return self.transform(img), torch.tensor(y1), torch.tensor(y2) #returns transformed image, and verdict an d source

#Class to store metrics for each training epoch 
@dataclass
class EpochResult:
    loss: float
    verdict_acc: float
    source_acc: float


#For running image training epochs  
def run_epoch(
    model: MobileNetV3DualHead, loader: DataLoader,criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> EpochResult:
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    verdict_correct = 0
    source_correct = 0
    total = 0

    #For btach runnung and calculating loss, accuracy
    for images, verdict_labels, source_labels in tqdm(loader, leave=False, mininterval=2.0):
        images = images.to(device)
        verdict_labels = verdict_labels.to(device)
        source_labels = source_labels.to(device)

        if train:
            optimizer.zero_grad()
    
        verdict_logits, source_logits = model(images)
        loss = criterion(verdict_logits, verdict_labels) + criterion(source_logits, source_labels)

        if train:
            loss.backward()
            optimizer.step()
#Caluclating performan metrics 
        total_loss += loss.item() * images.size(0)
        verdict_preds = verdict_logits.argmax(dim=1)
        source_preds = source_logits.argmax(dim=1)
        verdict_correct += (verdict_preds == verdict_labels).sum().item()
        source_correct += (source_preds == source_labels).sum().item()
        total += images.size(0)
    #returns all the associated metrics ofr each epoch 
    return EpochResult(
        loss=total_loss / max(total, 1),
        verdict_acc=verdict_correct / max(total, 1),
        source_acc=source_correct / max(total, 1),
    )

#Main function for training 
def main() -> None:
    parser = argparse.ArgumentParser(description="Train MobileNetV3 detector.")
    parser.add_argument("--dataset", default="Rajarshi-Roy-research/Defactify_Image_Dataset")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--step-size", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()
    #for lighter proccessing using CUDA or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #load train and validation datasets
    train_hf, img_k, l1_k, l2_k = load_hf_split(
        args.dataset, "train", args.max_train_samples
        )
    val_hf, _, _, _ = load_hf_split(
        args.dataset, "validation", args.max_val_samples
        )
 
    #create dataset and loaders 
    train_ds = DefactifyHFDataset(train_hf, 
    build_transforms(train=True), img_k, 
    l1_k, l2_k
    )
    
    val_ds = DefactifyHFDataset(val_hf, build_transforms(train=False), img_k, l1_k, l2_k)
   
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

    # Label column only 
    source_types = int(max(train_hf[l2_k])) + 1
    source_labels = SOURCES[:source_types]
    #Makes sure label mapping corresponds to dataset classes 
    if len(source_labels) < source_types:
        source_labels.extend([f"source_{i}" for i in range(len(source_labels), source_types)])
   #creates models with scheduling and learning rate 
    model = MobileNetV3DualHead(verdict_types=2, source_types=source_types).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    best_val_score = -1.0
    best_payload = None
    #Trains model for epoch amount 
    for epoch in range(1, args.epochs + 1):
        train_result = run_epoch(model, train_loader, criterion, device, optimizer)
        val_result = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()
        #calculates average accurayc of verdict and source 
        score = (val_result.verdict_acc + val_result.source_acc) / 2.0
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_result.loss:.4f}, train_verdict_acc={train_result.verdict_acc:.4f}, "
            f"train_source_acc={train_result.source_acc:.4f} | "
            f"val_loss={val_result.loss:.4f}, val_verdict_acc={val_result.verdict_acc:.4f}, "
            f"val_source_acc={val_result.source_acc:.4f}"
        )
        #saves best model for accuracy 
        if score > best_val_score:
            best_val_score = score
            best_payload = {
                "model_state_dict": model.state_dict(),
                "verdict_labels": VERDICTS,
                "source_labels": source_labels,
                "image_size": IMAGE_SIZE,
            }

    # Prevents using invalid checkpoint
    if best_payload is None:
        raise RuntimeError("Training couldn't to attempt create payload.")

    torch.save(best_payload, DEEP_CHECKPOINT_PATH)
    print(f"Saved best checkpoint to {DEEP_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
