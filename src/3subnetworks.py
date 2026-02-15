import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import open_clip
from tqdm import tqdm
from PIL import Image
import wandb
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

# =========================================================
# 1. W&B
# =========================================================
model_name = "3fcensemble_with_diversity_loss"
wandb.init(
    project="applied-dl-domain-adaptation",
    name=model_name,
)

# =========================================================
# 2. DEVICE
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================================================
# 3. LOAD CLIP
# =========================================================
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model = model.to(device)

# Freeze CLIP visual encoder
for param in model.visual.parameters():
    param.requires_grad = False
model.visual.eval()

feature_dim = model.visual.output_dim
print("CLIP feature dim:", feature_dim)

# =========================================================
# 4. DATASET
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

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
real_test_loader = DataLoader(real_test_dataset, batch_size=512, shuffle=False, num_workers=4)
infograph_test_loader = DataLoader(infograph_test_dataset, batch_size=512, shuffle=False, num_workers=4)
clipart_test_loader = DataLoader(clipart_test_dataset, batch_size=512, shuffle=False, num_workers=4)

# =========================================================
# 5. ENSEMBLE MODEL (3 subnetworks, 3 FC layers each)
# =========================================================
class FC_EnsembleMember(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.dropout(h)
        h = self.relu(self.fc2(h))
        h = self.dropout(h)
        logits = self.fc3(h)
        return logits, h  # return embedding for diversity


class CLIP_ThreeEnsembleMLP(nn.Module):
    def __init__(self, model, feature_dim, num_classes):
        super().__init__()
        self.model = model
        self.ensembles = nn.ModuleList([
            FC_EnsembleMember(feature_dim, num_classes)
            for _ in range(3)
        ])

    def forward(self, images):
        with torch.no_grad():
            features = self.model.encode_image(images)
            features = features / features.norm(dim=-1, keepdim=True) # normalize features

        logits_list = []
        embed_list = []

        for ensemble in self.ensembles:
            logits, h = ensemble(features)
            logits_list.append(logits)
            embed_list.append(h)

        return logits_list, embed_list


ensemble_model = CLIP_ThreeEnsembleMLP(
    model=model,
    feature_dim=feature_dim,
    num_classes=num_classes,
).to(device)

# =========================================================
# 6. LOSSES
# =========================================================
def compute_loss(logits_list, embed_list, labels,
                 lambda_div=0.2, alpha=0.1):

    # Mean logits for classification
    avg_logits = torch.mean(torch.stack(logits_list), dim=0)
    ce_loss = nn.CrossEntropyLoss()(avg_logits, labels)

    # Std-based diversity on embeddings
    Z = torch.stack(embed_list, dim=0)
    std = Z.std(dim=0)
    div_loss = F.relu(alpha - std.mean())

    total = ce_loss + lambda_div * div_loss

    return total, ce_loss, div_loss


# =========================================================
# 7. TRAINING
# =========================================================
def train_one_epoch(model, loader, optimizer, device, epoch):

    model.train()
    total_loss_val = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"Training Epoch {epoch+1}")):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits_list, embed_list = model(images)
        loss, ce_loss, div_loss = compute_loss(
            logits_list, embed_list, labels
        )

        loss.backward()
        optimizer.step()

        total_loss_val += loss.item()

        avg_logits = torch.mean(torch.stack(logits_list), dim=0)
        preds = avg_logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # LOG EVERY 10 STEPS
        if (batch_idx + 1) % 10 == 0:
            Z = torch.stack(embed_list, dim=0)
            current_std = Z.std(dim=0).mean().item()
            
            wandb.log({
                "train/train_loss": loss.item(),
                "train/train_acc_step": correct / total,
                "train/running_loss": total_loss_val / total,
                "train/step": batch_idx + 1,
                "train/embedding_std": current_std
            })
            print(
                f"[EPOCH {epoch+1}] "
                f"Step [{batch_idx+1}/{len(loader)}] "
                f"Batch Loss: {loss.item():.4f} "
                f"Train Acc: {correct/total:.4f} "
                f"Mean STD: {current_std:.4f}"
            )

    return total_loss_val / len(loader), correct / total


# =========================================================
# 8. EVALUATION
# =========================================================
@torch.no_grad()
def evaluate(model, loader, device):

    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        logits_list, _ = model(images)
        avg_logits = torch.mean(torch.stack(logits_list), dim=0)
        preds = avg_logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

    return acc, f1, precision, recall


# =========================================================
# 9. OPTIMIZER
# =========================================================
EPOCHS = 15

optimizer = torch.optim.Adam(
    ensemble_model.ensembles.parameters(),
    lr=1e-3,
    weight_decay=1e-5
)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5,gamma=0.1)


# =========================================================
# 10. TRAINING LOOP
# =========================================================
print("Training started...")

for epoch in range(EPOCHS):

    train_loss, train_acc = train_one_epoch(
        ensemble_model, train_loader, optimizer, device, epoch
    )

    scheduler.step()

    real_acc, real_f1, _, _ = evaluate(ensemble_model, real_test_loader, device)
    infograph_acc, infograph_f1, _, _ = evaluate(ensemble_model, infograph_test_loader, device)
    clipart_acc, clipart_f1, _, _ = evaluate(ensemble_model, clipart_test_loader, device)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Real Acc: {real_acc:.4f}")
    print(f"Infograph Acc: {infograph_acc:.4f}")
    print(f"Clipart Acc: {clipart_acc:.4f}")

    wandb.log({
        "epoch": epoch + 1,
        "train/loss": train_loss,
        "train/acc": train_acc,
        "real/acc": real_acc,
        "infograph/acc": infograph_acc,
        "clipart/acc": clipart_acc,
        "real/f1": real_f1,
        "infograph/f1": infograph_f1,
        "clipart/f1": clipart_f1,
    })
