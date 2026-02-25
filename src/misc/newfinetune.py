import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
from tqdm import tqdm
import open_clip
import wandb
import os

# =========================
# 1. W&B INIT
# =========================
wandb.init(
    project="applied-dl-domain-adaptation",
    name="finetune_clip_realImages_clean2",
)

# =========================
# 2. DEVICE
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# 3. LOAD CLIP
# =========================
print("STEP 1: Loading CLIP...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model = model.to(device)
print("STEP 1 COMPLETE")

# =========================
# 4. DATASET
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

# =========================
# 5. LOAD DATASETS
# =========================
print("STEP 2: Loading datasets...")

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
print("Number of classes:", num_classes)

print("STEP 2 COMPLETE")

# =========================
# 6. DATALOADERS
# =========================
print("STEP 3: Creating dataloaders...")

train_loader = DataLoader(
    train_dataset, batch_size=256, shuffle=True, num_workers=4
)
real_test_loader = DataLoader(
    real_test_dataset, batch_size=256, shuffle=False, num_workers=4
)
infograph_test_loader = DataLoader(
    infograph_test_dataset, batch_size=256, shuffle=False, num_workers=4
)

clipart_test_loader = DataLoader(
    clipart_test_dataset, batch_size=256, shuffle=False, num_workers=4
)

print("STEP 3 COMPLETE")

# =========================
# 7. CLASSIFIER HEAD
# =========================
image_feature_dim = model.visual.output_dim
classifier = nn.Linear(image_feature_dim, num_classes).to(device)

# =========================
# 8. LOSS & OPTIMIZER (FIXED)
# =========================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    list(model.visual.parameters()) + list(classifier.parameters()),
    lr=1e-3
)

# =========================
# 9. TRAINING FUNCTION
# =========================
def train_one_epoch(epoch):
    model.train()
    classifier.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Epoch {epoch+1} Training"
    ):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # CLIP image encoder
        image_features = model.encode_image(images)
        # normalization
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )

        logits = classifier(image_features)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

# =========================
# 10. EVALUATION FUNCTION
# =========================
@torch.no_grad()
def evaluate(loader):
    model.eval()
    classifier.eval()

    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )

        logits = classifier(image_features)
        _, preds = logits.max(1)

        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return correct / total

# =========================
# 11. TRAIN LOOP
# =========================
EPOCHS = 5

for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(epoch)

    real_acc = evaluate(real_test_loader)
    infograph_acc = evaluate(infograph_test_loader)
    clipart_acc = evaluate(clipart_test_loader)

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Train Acc:  {train_acc:.4f}")
    print(f"  Real Test Acc: {real_acc:.4f}")
    print(f"  Infograph Test Acc: {infograph_acc:.4f}")
    print(f"  Clipart Test Acc: {clipart_acc:.4f}")


    wandb.log({
        "epoch": epoch + 1,
        "train/loss": train_loss,
        "train/acc": train_acc,
        "test/real_acc": real_acc,
        "test/infograph_acc": infograph_acc,
        "test/clipart_acc": clipart_acc,
    })

# =========================
# 12. SAVE MODEL
# =========================
save_path = "clip_real_finetuned.pth"

torch.save(
    {
        "clip_visual_state_dict": model.visual.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "num_classes": num_classes,
    },
    save_path
)

print(f"\nModel saved to: {save_path}")
wandb.finish()
