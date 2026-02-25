import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset, random_split
import open_clip
import numpy as np
import os
from PIL import Image
import math
import itertools

# ----------------------------
# CONFIG
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
BATCH_SIZE = 32
LR_ADAPTER = 1e-3
DOMAIN_WEIGHT = 0.3
TEST_ON = "sketch"  # clipart / sketch / infograph

# ----------------------------
# Gradient Reversal Layer
# ----------------------------
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

def grl(x, lambda_):
    return GradientReversal.apply(x, lambda_)

# ----------------------------
# DANN with Similarity Head
# ----------------------------
class DANNAdapter(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, feature_dim)  # back to 512
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, features, text_features, lambda_=0.0):

        adapted = self.adapter(features)
        adapted = F.normalize(adapted, dim=-1)

        # cosine similarity logits
        logits = 100.0 * adapted @ text_features.T

        # domain branch
        rev = grl(adapted, lambda_)
        domain_logits = self.domain_classifier(rev)

        return logits, domain_logits

# ----------------------------
# Lambda schedule (standard DANN)
# ----------------------------
def dann_lambda(step, max_steps):
    p = step / max_steps
    return 2. / (1. + math.exp(-10 * p)) - 1

# ----------------------------
# Dataset Classes
# ----------------------------
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

# ----------------------------
# Load CLIP (Frozen)
# ----------------------------
print("Loading CLIP...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
clip_model = clip_model.to(DEVICE)

for p in clip_model.parameters():
    p.requires_grad = False

clip_model.eval()
print("CLIP loaded and frozen.")

feature_dim = clip_model.visual.output_dim

# ----------------------------
# Load Data
# ----------------------------
print("Loading datasets...")

real_train_dataset = DomainNetListDataset(
    "data/real", "data/real/real_train.txt", preprocess
)

real_test_dataset = DomainNetListDataset(
    "data/real", "data/real/real_test.txt", preprocess
)

val_size = len(real_test_dataset) // 2
real_val_dataset, real_test_dataset = random_split(
    real_test_dataset, [val_size, len(real_test_dataset) - val_size]
)

target_train_dataset = UnlabeledDomainDataset(
    f"data/{TEST_ON}",
    f"data/{TEST_ON}/{TEST_ON}_train.txt",
    preprocess,
)

target_test_dataset = DomainNetListDataset(
    f"data/{TEST_ON}",
    f"data/{TEST_ON}/{TEST_ON}_test.txt",
    preprocess,
)

val_size = len(target_test_dataset) // 2
target_val_dataset, target_test_dataset = random_split(
    target_test_dataset, [val_size, len(target_test_dataset) - val_size]
)

real_train_loader = DataLoader(real_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
target_train_loader = DataLoader(target_train_dataset, batch_size=BATCH_SIZE, shuffle=True)

real_val_loader = DataLoader(real_val_dataset, batch_size=BATCH_SIZE)
target_val_loader = DataLoader(target_val_dataset, batch_size=BATCH_SIZE)

num_classes = max(label for _, label in real_train_dataset.samples) + 1
print("Classes:", num_classes)

# ----------------------------
# Build Text Embeddings
# ----------------------------
print("Building text embeddings...")

# IMPORTANT: Replace with actual class names if available
class_names = [str(i) for i in range(num_classes)]
templates = ["a photo of a {}"]

texts = []
for cname in class_names:
    for template in templates:
        texts.append(template.format(cname))

tokenizer = open_clip.get_tokenizer("ViT-B-32")
text_tokens = tokenizer(texts).to(DEVICE)

with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens)
    text_features = F.normalize(text_features, dim=-1)

text_features = text_features.view(num_classes, len(templates), -1).mean(dim=1)
text_features = F.normalize(text_features, dim=-1)

# ----------------------------
# Initialize Model
# ----------------------------
dann = DANNAdapter(feature_dim).to(DEVICE)
optimizer = torch.optim.Adam(dann.parameters(), lr=LR_ADAPTER)
class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

# ----------------------------
# Evaluation Function
# ----------------------------
@torch.no_grad()
def evaluate(loader):
    dann.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        feats = clip_model.encode_image(images)
        feats = F.normalize(feats, dim=-1)

        logits, _ = dann(feats, text_features, lambda_=0.0)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total

# ----------------------------
# Training Loop
# ----------------------------
print("Starting training...")

total_steps = EPOCHS * min(len(real_train_loader), len(target_train_loader))
global_step = 0

for epoch in range(EPOCHS):

    dann.train()

    for (x_src, y_src), x_tgt in zip(real_train_loader, target_train_loader):

        x_src = x_src.to(DEVICE)
        y_src = y_src.to(DEVICE)
        x_tgt = x_tgt.to(DEVICE)

        lambda_ = dann_lambda(global_step, total_steps)

        with torch.no_grad():
            f_src = clip_model.encode_image(x_src)
            f_tgt = clip_model.encode_image(x_tgt)

        f_src = F.normalize(f_src, dim=-1)
        f_tgt = F.normalize(f_tgt, dim=-1)

        class_logits, dom_src = dann(f_src, text_features, lambda_)
        _, dom_tgt = dann(f_tgt, text_features, lambda_)

        class_loss = class_criterion(class_logits, y_src)

        dom_src_labels = torch.ones_like(dom_src)
        dom_tgt_labels = torch.zeros_like(dom_tgt)

        domain_loss = (
            domain_criterion(dom_src, dom_src_labels)
            + domain_criterion(dom_tgt, dom_tgt_labels)
        )

        loss = class_loss + DOMAIN_WEIGHT * domain_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1

    real_acc = evaluate(real_val_loader)
    target_acc = evaluate(target_val_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss {loss.item():.4f} | "
        f"Real {real_acc:.4f} | "
        f"Target {target_acc:.4f}"
    )

print("Training complete.")