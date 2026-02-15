import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import open_clip
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score
import wandb


# =========================================================
# 1. W&B
# =========================================================
model_name = "3fcensemble_corrected_without_diversity_loss"
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

# Freeze CLIP visual encoder
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
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        return self.net(x)


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
            features = features / features.norm(dim=-1, keepdim=True)

        logits_list = [ensemble(features) for ensemble in self.ensembles]
        return logits_list


ensemble_model = CLIP_ThreeEnsembleMLP(
    model=model,
    feature_dim=feature_dim,
    num_classes=num_classes,
).to(device)

# =========================================================
# 6. LOSS (Mean logits only, NO diversity)
# =========================================================
criterion = nn.CrossEntropyLoss()

def compute_loss(logits_list, labels):
    avg_logits = torch.mean(torch.stack(logits_list), dim=0)
    return criterion(avg_logits, labels)

# =========================================================
# 7. TRAINING
# =========================================================
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="Training")):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits_list = model(images)
        loss = compute_loss(logits_list, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        avg_logits = torch.mean(torch.stack(logits_list), dim=0)
        preds = avg_logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # LOG EVERY 10 STEPS
        if (batch_idx + 1) % 10 == 0:
            wandb.log({
                "train/train_loss": loss.item(),
                "train/train_acc_step": correct / total,
                "train/running_loss": total_loss / total,
                "train/step": batch_idx + 1
            })
            print(
                f"Step [{batch_idx+1}/{len(loader)}] "
                f"Batch Loss: {loss.item():.4f} "
                f"Train Acc: {correct/total:.4f}"
            )

    return total_loss / len(loader), correct / total


# =========================================================
# 8. EVALUATION
# =========================================================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits_list = model(images)
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

    return acc, f1


# =========================================================
# 9. OPTIMIZER
# =========================================================
EPOCHS = 15

optimizer = optim.Adam(
    ensemble_model.ensembles.parameters(),
    lr=1e-2,
    weight_decay=1e-5
)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=EPOCHS
# )
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5,gamma=0.1)

# =========================================================
# 10. TRAIN LOOP
# =========================================================
print("Training started...")

for epoch in range(EPOCHS):

    train_loss, train_acc = train_one_epoch(
        ensemble_model, train_loader, optimizer
    )

    scheduler.step()

    real_acc, real_f1 = evaluate(ensemble_model, real_test_loader)
    infograph_acc, infograph_f1 = evaluate(ensemble_model, infograph_test_loader)
    clipart_acc, clipart_f1 = evaluate(ensemble_model, clipart_test_loader)

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

# =========================================================
# 11. SAVE MODEL
# =========================================================
# os.makedirs("models", exist_ok=True)

# torch.save(
#     {
#         "ensemble_state_dict": ensemble_model.state_dict(),
#         "num_classes": num_classes,
#     },
#     "models/clip_3ensemble_no_diversity.pth"
# )

# print("\nModel saved to models/clip_3ensemble_no_diversity.pth")
