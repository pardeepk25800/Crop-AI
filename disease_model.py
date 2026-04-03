# disease_model.py — CNN Disease Detection (EfficientNet-B3 + Transfer Learning)

import os
import json
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

from config import (
    DISEASE_DATA_DIR, MODEL_DIR, RESULTS_DIR, DATA_DIR,
    IMAGE_SIZE, BATCH_SIZE, EPOCHS, LR, LR_PATIENCE,
    DROPOUT, TRAIN_SPLIT, MEAN, STD, NUM_CLASSES,
    DISEASE_MODEL_PATH, BACKBONE, NUM_WORKERS,
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DiseaseModel] Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATASET
# ─────────────────────────────────────────────────────────────────────────────

class LeafDataset(Dataset):
    """Loads leaf images from folder structure: root/ClassName/img.jpg"""

    def __init__(self, root_dir: str, transform=None):
        self.root_dir  = root_dir
        self.transform = transform
        self.samples   = []  # list of (path, class_idx)
        self.classes   = []

        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls])
                    )

        print(f"[LeafDataset] {len(self.samples)} images | {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])


def build_dataloaders(root_dir: str):
    full_ds = LeafDataset(root_dir, transform=get_transforms(train=True))
    val_ds  = LeafDataset(root_dir, transform=get_transforms(train=False))

    n      = len(full_ds)
    if n < 3:
        # Fallback for tiny datasets used in demos
        n_tr, n_val, n_te = n, 0, 0
        if n == 2: n_tr, n_val = 1, 1
    else:
        n_val = max(1, int(n * 0.1))
        n_te  = max(1, int(n * 0.1))
        n_tr  = n - n_val - n_te

    train_idx, val_idx, test_idx = random_split(
        range(n), [n_tr, n_val, n_te],
        generator=torch.Generator().manual_seed(42)
    )

    from torch.utils.data import Subset
    train_loader = DataLoader(
        Subset(full_ds, train_idx), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        Subset(val_ds, test_idx), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    print(f"[DataLoader] Train={len(train_idx)} | Val={len(val_idx)} | Test={len(test_idx)}")
    return train_loader, val_loader, test_loader, full_ds.classes


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

class CropDiseaseModel(nn.Module):
    """
    Transfer learning on EfficientNet-B3 with a custom classification head.
    Can swap backbone to 'resnet50' or 'mobilenet_v3_large'.
    """

    def __init__(self, num_classes: int, backbone: str = BACKBONE,
                 dropout: float = DROPOUT, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone
        weights = "IMAGENET1K_V1" if pretrained else None

        if backbone == "efficientnet_b3":
            base = models.efficientnet_b3(weights=weights)
            in_features = base.classifier[1].in_features
            base.classifier = nn.Identity()

        elif backbone == "resnet50":
            base = models.resnet50(weights=weights)
            in_features = base.fc.in_features
            base.fc = nn.Identity()

        elif backbone == "mobilenet_v3_large":
            base = models.mobilenet_v3_large(weights=weights)
            in_features = base.classifier[3].in_features
            base.classifier = nn.Identity()

        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.backbone = base
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.flatten(1)
        return self.head(features)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self, layers_from_end: int = 2):
        """Unfreeze last N blocks for fine-tuning."""
        if hasattr(self.backbone, "features"):
            # EfficientNet, MobileNet etc. have 'features' sequential blocks
            blocks = list(self.backbone.features.children())
            for block in blocks[-layers_from_end:]:
                for p in block.parameters():
                    p.requires_grad = True
        else:
            # ResNet etc. have children directly. Filter to children with parameters.
            param_modules = [m for m in self.backbone.children() if any(p.requires_grad is not None for p in m.parameters())]
            for m in param_modules[-layers_from_end:]:
                for p in m.parameters():
                    p.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def train_model(root_dir: str = DISEASE_DATA_DIR):
    print("\n" + "=" * 60)
    print("  CropAI — Disease Detection Training")
    print("=" * 60)

    train_loader, val_loader, test_loader, classes = build_dataloaders(root_dir)
    num_cls = len(classes)
    print(f"[Train] Classes: {num_cls}")

    # Save class mapping
    mapping = {i: c for i, c in enumerate(classes)}
    with open(os.path.join(DATA_DIR, "class_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)

    model = CropDiseaseModel(num_classes=num_cls).to(DEVICE)
    model.freeze_backbone()     # Phase 1: train head only

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=0.5
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, EPOCHS + 1):
        # Phase 2: unfreeze backbone after epoch 5
        if epoch == 6:
            model.unfreeze_backbone(layers_from_end=3)
            for pg in optimizer.param_groups:
                pg["lr"] = LR / 10
            print("[Train] Backbone unfrozen — fine-tuning with lr/10")

        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step(vl_loss)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(f"  Epoch {epoch:02d}/{EPOCHS} | "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc*100:.2f}% | "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc*100:.2f}% | "
              f"{elapsed:.1f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_weights = copy.deepcopy(model.state_dict())
            print(f"  [OK] New best val accuracy: {best_val_acc*100:.2f}%")

    # Restore best weights
    model.load_state_dict(best_weights)

    # Test evaluation
    te_loss, te_acc = eval_epoch(model, test_loader, criterion, DEVICE)
    print(f"\n[Train] [OK] Test Accuracy: {te_acc*100:.2f}%")

    # Save model
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes":          classes,
        "num_classes":      num_cls,
        "backbone":         BACKBONE,
        "test_accuracy":    te_acc,
        "history":          history,
    }, DISEASE_MODEL_PATH)
    print(f"[Train] [OK] Model saved -> {DISEASE_MODEL_PATH}")

    # Plot training curves
    _plot_history(history)
    return model, classes, history


def _plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot([a * 100 for a in history["train_acc"]], label="Train")
    axes[1].plot([a * 100 for a in history["val_acc"]],   label="Val")
    axes[1].set_title("Accuracy (%)"); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "disease_training_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[Train] [OK] Training curves saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

class DiseasePredictor:
    """Load a trained model and run inference on single images."""

    def __init__(self, model_path: str = DISEASE_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                "Run train_model() first or call python disease_model.py"
            )
        checkpoint = torch.load(model_path, map_location=DEVICE)
        self.classes     = checkpoint["classes"]
        self.num_classes = checkpoint["num_classes"]
        backbone         = checkpoint.get("backbone", BACKBONE)

        self.model = CropDiseaseModel(
            num_classes=self.num_classes, backbone=backbone
        ).to(DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.transform = get_transforms(train=False)
        print(f"[DiseasePredictor] Model loaded - {self.num_classes} classes")

    @torch.no_grad()
    def predict(self, image_input, top_k: int = 3):
        """
        image_input: PIL.Image, file path (str), or raw bytes
        Returns dict with prediction details.
        """
        if isinstance(image_input, (str, bytes)):
            if isinstance(image_input, str):
                img = Image.open(image_input).convert("RGB")
            else:
                from io import BytesIO
                img = Image.open(BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise TypeError("Expected PIL.Image, file path, or bytes")

        tensor = self.transform(img).unsqueeze(0).to(DEVICE)
        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

        top_probs, top_idxs = probs.topk(min(top_k, self.num_classes))

        top_preds = [
            {
                "class":      self.classes[idx.item()],
                "confidence": round(prob.item() * 100, 2),
            }
            for prob, idx in zip(top_probs, top_idxs)
        ]

        primary = top_preds[0]
        cls_name = primary["class"]

        return {
            "predicted_class": cls_name,
            "confidence":      primary["confidence"],
            "is_healthy":      "healthy" in cls_name.lower(),
            "crop":            cls_name.split("___")[0].replace("_", " "),
            "disease":         cls_name.split("___")[-1].replace("_", " "),
            "top_predictions": top_preds,
        }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick test with synthetic data
    print("Checking for dataset …")
    if not os.path.isdir(DISEASE_DATA_DIR):
        print("Dataset not found — generating synthetic data …")
        from data_generator import generate_disease_dataset
        generate_disease_dataset(samples_per_class=30)

    model, classes, history = train_model()

    print("\n[Test] Running inference on a random image …")
    predictor = DiseasePredictor()
    test_img = Image.new("RGB", (224, 224), color=(34, 139, 34))
    result = predictor.predict(test_img)
    print(json.dumps(result, indent=2))
