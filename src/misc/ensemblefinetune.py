import os
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import open_clip
from tqdm import tqdm
from PIL import Image
import wandb
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


# wandb initialize
model_name = "3fcensemblefinetunerun(lr=0.01,lambda=0.05)-2layer-cosineSched"
wandb.init(
    project="applied-dl-domain-adaptation",
    name=model_name,
)
# =========================================================
# 1. DEVICE
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================================================
# 2. LOAD CLIP
# =========================================================
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model = model.to(device)
print("STEP 1 COMPLETE")

# ---- Freeze CLIP visual encoder ----
for param in model.visual.parameters():
    param.requires_grad = False
model.visual.eval()

feature_dim = model.visual.output_dim
print("CLIP feature dim:", feature_dim)

# =========================================================
# 3. DATASET
# =========================================================
class DomainNetListDataset(Dataset):
    def __init__(self, root_dir, txt_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        with open(txt_file, "r") as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue
                if len(tokens) < 2:
                    raise ValueError(f"Malformed line in {txt_file}: {line!r}")
                path, label = tokens[0], int(tokens[-1])
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, rel_path)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

# =========================================================
# 4. LOAD DATA
# =========================================================
train_dataset = DomainNetListDataset(
    root_dir="data/real",
    txt_file="data/real/real_train.txt",
    transform=preprocess,
)

real_test_dataset = DomainNetListDataset(
    root_dir="data/real",
    txt_file="data/real/real_test.txt",
    transform=preprocess,
)

infograph_test_dataset = DomainNetListDataset(
    root_dir="data/infograph",
    txt_file="data/infograph/infograph_test.txt",
    transform=preprocess,
)

clipart_test_dataset = DomainNetListDataset(
    root_dir="data/clipart",
    txt_file="data/clipart/clipart_test.txt",
    transform=preprocess,
)

num_classes = max(label for _, label in train_dataset.samples) + 1
print("Num classes:", num_classes)
print("Step 2 complete")

print("Creating data loaders...")
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4)
real_test_loader = DataLoader(real_test_dataset, batch_size=1024, shuffle=False, num_workers=4)
infograph_test_loader = DataLoader(infograph_test_dataset, batch_size=1024, shuffle=False, num_workers=4)
clipart_test_loader = DataLoader(clipart_test_dataset, batch_size=1024, shuffle=False, num_workers=4)
print("Step 3 complete")

# =========================================================
# 5. ENSEMBLE MODEL (3 MLPs, each 3 FC layers)
# =========================================================
class FC_EnsembleMember(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            # nn.Linear(1024, 1024),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.1),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CLIP_ThreeEnsembleMLP(nn.Module):
    def __init__(self, model, feature_dim, num_classes, num_ensembles=3):
        super().__init__()
        self.model = model
        self.ensembles = nn.ModuleList([
            FC_EnsembleMember(feature_dim, num_classes)
            for _ in range(num_ensembles)
        ])

    def forward(self, images):
        with torch.no_grad():  # CLIP is frozen
            features = self.model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True)

        logits = [ensemble(features) for ensemble in self.ensembles]
        return logits, features


ensemble_model = CLIP_ThreeEnsembleMLP(
    model=model,
    feature_dim=feature_dim,
    num_classes=num_classes,
).to(device)

# =========================================================
# 6. LOSSES
# =========================================================
def classification_loss(logits_list, labels):
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    for logits in logits_list:
        loss += criterion(logits, labels)
    return loss / max(1, len(logits_list))


def diversity_loss(logits_list):
    loss = 0.0
    num_pairs = 0
    for i in range(len(logits_list)):
        for j in range(i + 1, len(logits_list)):
            zi = F.normalize(logits_list[i], dim=1)
            zj = F.normalize(logits_list[j], dim=1)
            loss += (zi * zj).sum(dim=1).mean()
            num_pairs += 1
    if num_pairs == 0:
        return 0.0
    return loss / num_pairs


def total_loss(logits_list, labels, lambda_div=0.1):
    return classification_loss(logits_list, labels) + lambda_div * diversity_loss(logits_list)

# =========================================================
# 8. TRAINING
# =========================================================
def train_one_epoch(model, loader, optimizer, device, lambda_div=0.1, epoch=None):
    model.train()
    total_loss_val = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Training")):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits_list, _ = model(images)
        loss = total_loss(logits_list, labels, lambda_div)
        loss.backward()
        optimizer.step()

        total_loss_val += loss.item()

        avg_logits = torch.mean(torch.stack(logits_list), dim=0)
        preds = avg_logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)


        # LOG EVERY 10 STEPS - you can change this 10 to any number you like
        if (batch_idx + 1) % 10 == 0:
            wandb.log({
                "train/train_loss": loss.item(),
                "train/train_acc_step": correct / total if total > 0 else 0.0,
                "train/running_loss": total_loss_val / (batch_idx + 1),
                "train/step": batch_idx + 1
            })
            info_epoch = f"[EPOCH {epoch}]:" if epoch is not None else ""
            print(
                f"{info_epoch}"
                f"Train Step [{batch_idx+1}/{len(loader)}] "
                f"Batch Loss: {loss.item():.4f} "
                f"Train Acc: {correct/total:.4f}"
            )

    return total_loss_val / len(loader), correct / total

# =========================================================
# 9. EVALUATION
# =========================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits_list, _ = model(images)
        avg_logits = torch.mean(torch.stack(logits_list), dim=0)
        preds = avg_logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Accumulate predictions and labels for metric computation
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    #compute metrics over entire dataset
    if len(all_labels) == 0:
        # empty loader or no samples
        return 0.0, 0.0, 0.0, 0.0

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    acc = correct / total if total > 0 else 0.0
    f1 = f1_score(all_labels, all_preds, zero_division=0, average='weighted')
    precision = precision_score(all_labels, all_preds, zero_division=0, average='weighted')
    recall = recall_score(all_labels, all_preds, zero_division=0, average='weighted')


    return acc, f1, precision, recall

# =========================================================
# 10. RUN TRAINING
# =========================================================
print("Training has started...")

EPOCHS = 15
LAMBDA_DIV = 0.05
optimizer = torch.optim.Adam(
    ensemble_model.ensembles.parameters(),
    lr=0.01, weight_decay=1e-5
)
# scheduler = optim.lr_scheduler.StepLR(
#     optimizer,
#     step_size=5,
#     gamma=0.1
# )

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(
        ensemble_model, train_loader, optimizer, device, LAMBDA_DIV
    )

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Acc: {train_acc:.4f}")
    
    # Evaluate on test sets: real, infograph, clipart
    real_acc, real_f1, real_precision, real_recall = evaluate(ensemble_model, real_test_loader, device)
    infograph_acc, infograph_f1, infograph_precision, infograph_recall = evaluate(ensemble_model, infograph_test_loader, device)
    clipart_acc, clipart_f1, clipart_precision, clipart_recall = evaluate(ensemble_model, clipart_test_loader, device)

    print(f"Real Test Acc: {real_acc:.4f}")
    print(f"Real Test F1: {real_f1:.4f}")
    print(f"Real Test Precision: {real_precision:.4f}")
    print(f"Real Test Recall: {real_recall:.4f}")

    print(f"Infograph Test Acc: {infograph_acc:.4f}")
    print(f"Infograph Test F1: {infograph_f1:.4f}")
    print(f"Infograph Test Precision: {infograph_precision:.4f}")
    print(f"Infograph Test Recall: {infograph_recall:.4f}")

    print(f"Clipart Test Acc: {clipart_acc:.4f}")
    print(f"Clipart Test F1: {clipart_f1:.4f}")
    print(f"Clipart Test Precision: {clipart_precision:.4f}")
    print(f"Clipart Test Recall: {clipart_recall:.4f}")

    wandb.log({
        "epoch": epoch + 1,
        "train/loss": train_loss,
        "train/acc": train_acc,
        "test/real_acc": real_acc,
        "test/infograph_acc": infograph_acc,
        "test/clipart_acc": clipart_acc,

        "test/real_f1": real_f1,
        "test/infograph_f1": infograph_f1,
        "test/clipart_f1": clipart_f1,

        "test/real_precision": real_precision,
        "test/infograph_precision": infograph_precision,
        "test/clipart_precision": clipart_precision,

        "test/real_recall": real_recall,
        "test/infograph_recall": infograph_recall,
        "test/clipart_recall": clipart_recall,
    })

    # step LR scheduler
    if 'scheduler' in globals():
        try:
            scheduler.step()
        except Exception:
            pass


# =========================================================
# 11. SAVE MODEL
# =========================================================
#os.makedirs("checkpoints", exist_ok=True)

# torch.save(
#     {
#         "ensemble_state_dict": ensemble_model.ensembles.state_dict(),
#         "num_classes": num_classes,
#     },
#     "models/clip_3ensemble_mlp_lr0.01.pth"
# )

# print("\nModel saved to models/clip_3ensemble_mlp_lr0.01.pth")