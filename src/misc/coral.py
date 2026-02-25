# ============================================================
# CORAL Domain Adaptation with CLIP (DomainNet)
# Source: Real (labeled)  -> Target: {sketch|clipart|infograph} (unlabeled)
# Works for METHOD = "frozen" (recommended) or "unfrozen"
# ============================================================

import os
import math
import itertools
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

import open_clip
import wandb
import torch.optim as optim


# =========================
# Config
# =========================
METHOD = "frozen"       # "frozen" or "unfrozen"
TEST_ON = "sketch"      # "sketch" or "clipart" or "infograph"
SCHEDULER = "cosine"    # "cosine" or "stepLR"

epochs = 20
BATCH_SIZE = 64

LR_clip = 1e-6          # only used if METHOD="unfrozen"
LR_head = 1e-3

alpha_coral = 0.5       # weight for CORAL loss (tune: 0.1 -> 1.0)
weight_decay = 1e-5
num_workers = 4
seed = 42


# =========================
# Reproducibility
# =========================
torch.manual_seed(seed)
np.random.seed(seed)


# =========================
# W&B init
# =========================
run_name = f"real_{TEST_ON}_CLIP{METHOD}_CORAL_lrclip{LR_clip}_lrhead{LR_head}_{SCHEDULER}_a{alpha_coral}_bs{BATCH_SIZE}"
wandb.init(project="applied-dl-domain-adaptation", name=run_name)


# =========================
# Dataset classes
# =========================
class DomainNetListDataset(Dataset):
    def __init__(self, root_dir, txt_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        with open(txt_file, "r") as f:
            for line in f:
                path, label = line.strip().split()
                self.samples.append((path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, rel_path)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class UnlabeledDomainDataset(Dataset):
    def __init__(self, root_dir, txt_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        with open(txt_file, "r") as f:
            for line in f:
                path, _ = line.strip().split()
                self.samples.append(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path = self.samples[idx]
        img_path = os.path.join(self.root_dir, rel_path)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# =========================
# CORAL loss
# =========================
def coral_loss(source: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    CORAL loss between source and target features.

    source: (Ns, D)
    target: (Nt, D)
    """
    assert source.dim() == 2 and target.dim() == 2
    d = source.size(1)

    # Center features
    source = source - source.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)

    # Covariance
    ns = max(source.size(0) - 1, 1)
    nt = max(target.size(0) - 1, 1)

    cov_s = (source.t() @ source) / ns
    cov_t = (target.t() @ target) / nt

    # Optional tiny ridge for numerical stability
    cov_s = cov_s + eps * torch.eye(d, device=source.device, dtype=source.dtype)
    cov_t = cov_t + eps * torch.eye(d, device=target.device, dtype=target.dtype)

    loss = ((cov_s - cov_t) ** 2).sum() / (4.0 * d * d)
    return loss


# =========================
# Head / Adapter (like your DANN adapter+classifier without domain discriminator)
# =========================
class CORALHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 512):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor):
        z = self.adapter(x)
        logits = self.classifier(z)
        return logits, z  # return adapted features too


# =========================
# Load CLIP
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
clip_model = clip_model.to(device)

if METHOD == "frozen":
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_model.eval()
else:
    for p in clip_model.parameters():
        p.requires_grad = True
    clip_model.train()


# =========================
# Data loading (same structure as your DANN script)
# =========================
if TEST_ON == "infograph":
    datapath_name = "infograph"
elif TEST_ON == "clipart":
    datapath_name = "clipart"
elif TEST_ON == "sketch":
    datapath_name = "sketch"
else:
    raise ValueError("TEST_ON must be one of: infograph, clipart, sketch")

print("Loading datasets...")

real_train_dataset = DomainNetListDataset(
    root_dir="data/real",
    txt_file="data/real/real_train.txt",
    transform=preprocess,
)

real_testing_dataset = DomainNetListDataset(
    root_dir="data/real",
    txt_file="data/real/real_test.txt",
    transform=preprocess,
)

val_size = int(len(real_testing_dataset) * 0.5)
test_size = len(real_testing_dataset) - val_size
real_val_dataset, real_test_dataset = random_split(
    real_testing_dataset,
    [val_size, test_size],
    generator=torch.Generator().manual_seed(seed)
)

target_train_dataset = UnlabeledDomainDataset(
    root_dir=f"data/{datapath_name}",
    txt_file=f"data/{datapath_name}/{datapath_name}_train.txt",
    transform=preprocess,
)

target_testing_dataset = DomainNetListDataset(
    root_dir=f"data/{datapath_name}",
    txt_file=f"data/{datapath_name}/{datapath_name}_test.txt",
    transform=preprocess,
)

val_size = int(len(target_testing_dataset) * 0.5)
test_size = len(target_testing_dataset) - val_size
target_val_dataset, target_test_dataset = random_split(
    target_testing_dataset,
    [val_size, test_size],
    generator=torch.Generator().manual_seed(seed)
)

num_classes = max(label for _, label in real_train_dataset.samples) + 1
print(f"Source train: {len(real_train_dataset)} | classes: {num_classes}")
print(f"Target train (unlabeled): {len(target_train_dataset)}")

real_train_loader = DataLoader(real_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
real_val_loader   = DataLoader(real_val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
real_test_loader  = DataLoader(real_test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

target_train_loader = DataLoader(target_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
target_val_loader   = DataLoader(target_val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
target_test_loader  = DataLoader(target_test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)


# =========================
# Model + optimizer
# =========================
feature_dim = clip_model.visual.output_dim
head = CORALHead(feature_dim, num_classes, hidden_dim=512).to(device)

if METHOD == "frozen":
    optimizer = torch.optim.Adam(head.parameters(), lr=LR_head, weight_decay=weight_decay)
else:
    optimizer = torch.optim.Adam(
        [{"params": clip_model.parameters(), "lr": LR_clip},
         {"params": head.parameters(), "lr": LR_head}],
        weight_decay=weight_decay
    )

if SCHEDULER == "stepLR":
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
elif SCHEDULER == "cosine":
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
else:
    raise ValueError("SCHEDULER must be 'stepLR' or 'cosine'")

class_criterion = nn.CrossEntropyLoss()


# =========================
# Evaluation
# =========================
@torch.no_grad()
def evaluate_accuracy(clip_model, head, loader):
    clip_model.eval()
    head.eval()
    correct, total = 0, 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        feats = clip_model.encode_image(images).float()
        # Keep consistent: normalize only if you want (optional).
        # If you normalize, normalize in BOTH train and eval.
        # feats = F.normalize(feats, dim=-1)

        logits, _ = head(feats)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / max(total, 1)


# =========================
# Training loop (CORAL)
# =========================
print("Starting CORAL training...")
global_step = 0
steps_per_epoch = min(len(real_train_loader), len(target_train_loader))

for epoch in range(epochs):
    head.train()
    if METHOD == "unfrozen":
        clip_model.train()
    else:
        clip_model.eval()

    epoch_loss = 0.0
    epoch_cls = 0.0
    epoch_coral = 0.0

    for (x_src, y_src), x_tgt in itertools.islice(zip(real_train_loader, target_train_loader), steps_per_epoch):
        x_src = x_src.to(device)
        y_src = y_src.to(device)
        x_tgt = x_tgt.to(device)

        # Extract CLIP features
        if METHOD == "frozen":
            with torch.no_grad():
                f_src = clip_model.encode_image(x_src).float()
                f_tgt = clip_model.encode_image(x_tgt).float()
        else:
            f_src = clip_model.encode_image(x_src).float()
            f_tgt = clip_model.encode_image(x_tgt).float()

        # Optional normalization (if you enable, do it everywhere)
        # f_src = F.normalize(f_src, dim=-1)
        # f_tgt = F.normalize(f_tgt, dim=-1)

        # Forward
        logits_src, z_src = head(f_src)
        _, z_tgt = head(f_tgt)

        # Losses
        cls_loss = class_criterion(logits_src, y_src)
        c_loss = coral_loss(z_src, z_tgt)

        loss = cls_loss + alpha_coral * c_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_cls += cls_loss.item()
        epoch_coral += c_loss.item()

        if global_step % 10 == 0:
            wandb.log({
                "step": global_step,
                "epoch": epoch,
                "loss": loss.item(),
                "class_loss": cls_loss.item(),
                "coral_loss": c_loss.item(),
                "alpha_coral": alpha_coral,
            })

        global_step += 1

    # Epoch averages
    denom = max(steps_per_epoch, 1)
    epoch_loss /= denom
    epoch_cls /= denom
    epoch_coral /= denom

    # Validation
    real_val_acc = evaluate_accuracy(clip_model, head, real_val_loader)
    target_val_acc = evaluate_accuracy(clip_model, head, target_val_loader)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"loss={epoch_loss:.4f} (cls={epoch_cls:.4f}, coral={epoch_coral:.4f}) | "
        f"real_val_acc={real_val_acc:.4f} | target_val_acc={target_val_acc:.4f}"
    )

    wandb.log({
        "epoch": epoch,
        "epoch_loss": epoch_loss,
        "epoch_class_loss": epoch_cls,
        "epoch_coral_loss": epoch_coral,
        "real_val_acc": real_val_acc,
        "target_val_acc": target_val_acc,
    })

    scheduler.step()


# =========================
# Final test
# =========================
print("\n" + "=" * 50)
print("Final evaluation after CORAL adaptation")
print("=" * 50)

real_test_acc = evaluate_accuracy(clip_model, head, real_test_loader)
target_test_acc = evaluate_accuracy(clip_model, head, target_test_loader)

print(f"Real Test Accuracy:   {real_test_acc:.4f}")
print(f"Target Test Accuracy: {target_test_acc:.4f}")

wandb.log({
    "final/real_test_accuracy": real_test_acc,
    "final/target_test_accuracy": target_test_acc,
})

wandb.finish()