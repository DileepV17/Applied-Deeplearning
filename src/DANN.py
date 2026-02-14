import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import os
from PIL import Image
import wandb

import open_clip

# wandb initialize
wandb.init(
    project="applied-dl-domain-adaptation",
    name="Implement_DANN",
)

# gradient reversal layer
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

#DANN head (feature level adaption)
class DANNHead(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden_dim=512):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.class_classifier = nn.Linear(hidden_dim, num_classes)

        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, features, lambda_=0.0):
        features = self.adapter(features)

        class_logits = self.class_classifier(features)

        rev_features = grl(features, lambda_)
        domain_logits = self.domain_classifier(rev_features)

        return class_logits, domain_logits

    def extract_features(self, features):
        return self.adapter(features)
# lambda schedule (standard DANN)

def dann_lambda(step, max_steps):
    p = step / max_steps
    return 2. / (1. + math.exp(-10 * p)) - 1

# Dataset classes
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
    """Dataset that returns only images (no labels) for unsupervised domain adaptation"""
    def __init__(self, root_dir, txt_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        with open(txt_file, "r") as f:
            for line in f:
                path, _ = line.strip().split()  # Ignore label
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

#load clip
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP base model
print("Loading CLIP base model...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
clip_model = clip_model.to(device)

# Load fine-tuned weights
print("Loading fine-tuned weights...")
checkpoint = torch.load("clip_real_finetuned.pth", map_location=device)
clip_model.visual.load_state_dict(checkpoint["clip_visual_state_dict"])
clip_model.eval()
print("Model loaded.")

## Datasets placeholders

# source_loader: (image, label) - labeled source domain
# target_loader: (image) - unlabeled target domain for adaptation

# Load Datasets
print("Loading Datasets...")

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

# Target domain: UNLABELED for adaptation
target_dataset = UnlabeledDomainDataset(
    root_dir="data/infograph",
    txt_file="data/infograph/infograph_train.txt",
    transform=preprocess,
)

# Also load labeled infograph TEST dataset for evaluation (held-out)
infograph_eval_dataset = DomainNetListDataset(
    root_dir="data/infograph",
    txt_file="data/infograph/infograph_test.txt",
    transform=preprocess,
)

num_classes = max(label for _, label in train_dataset.samples) + 1
print(f"Loaded {len(train_dataset)} training samples, {num_classes} classes")
print("Datasets loaded.")

print("Preparing DataLoaders...")
source_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4)
target_loader = DataLoader(target_dataset, batch_size=512, shuffle=True, num_workers=4)
real_test_loader = DataLoader(real_test_dataset, batch_size=512, shuffle=False, num_workers=4)
infograph_eval_loader = DataLoader(infograph_eval_dataset, batch_size=512, shuffle=False, num_workers=4)
print("DataLoaders ready.")

#7. Initialize DANN

print(f"Initializing DANN with {num_classes} classes...")
feature_dim = clip_model.visual.output_dim

dann = DANNHead(feature_dim, num_classes).to(device)

optimizer = torch.optim.Adam(
    dann.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

#8. Accuracy evaluation function
@torch.no_grad()
def evaluate_accuracy(clip_model, dann, loader):
    """Evaluate accuracy using DANN-adapted features and class classifier"""
    dann.eval()
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Get CLIP features
        clip_feats = clip_model.encode_image(images).float()
        
        # Get adapted features and class logits
        class_logits, _ = dann(clip_feats, lambda_=0.0)
        
        # Get predictions
        _, preds = class_logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    
    accuracy = correct / total
    return accuracy

#9. t-sne helper function
def extract_tsne_features(clip_model, dann, loader, domain_label, max_batches=20):
    features = []
    domains = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break

            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)

            clip_feats = clip_model.encode_image(images).float()
            adapted_feats = dann.extract_features(clip_feats)

            features.append(adapted_feats.cpu().numpy())
            domains.append(
                np.full(len(images), domain_label)
            )

    return np.vstack(features), np.hstack(domains)
#9. t-sne before adaption

print("Extracting features BEFORE adaptation...")

src_feats, src_dom = extract_tsne_features(
    clip_model, dann, source_loader, domain_label=1
)
tgt_feats, tgt_dom = extract_tsne_features(
    clip_model, dann, target_loader, domain_label=0
)

X = np.vstack([src_feats, tgt_feats])
y = np.hstack([src_dom, tgt_dom])

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X)

plt.figure(figsize=(6, 6))
plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], label="Source", alpha=0.6)
plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], label="Target", alpha=0.6)
plt.legend()
plt.title("t-SNE BEFORE Domain Adaptation")
plt.savefig("tsne_before_adaptation.png", dpi=150, bbox_inches='tight')
wandb.log({"tsne_before_adaptation": wandb.Image("tsne_before_adaptation.png")})
plt.show()

#10. DANN training loop
print("Starting DANN training...")
epochs = 10
total_steps = epochs * min(len(source_loader), len(target_loader))
global_step = 0

for epoch in range(epochs):
    dann.train()

    for (x_src, y_src), x_tgt in zip(source_loader, target_loader):
        x_src = x_src.to(device)
        y_src = y_src.to(device)
        x_tgt = x_tgt.to(device)

        lambda_ = dann_lambda(global_step, total_steps)

        with torch.no_grad():
            f_src = clip_model.encode_image(x_src).float()
            f_tgt = clip_model.encode_image(x_tgt).float()

        class_logits, dom_logits_src = dann(f_src, lambda_)
        _, dom_logits_tgt = dann(f_tgt, lambda_)

        class_loss = class_criterion(class_logits, y_src)

        dom_src_labels = torch.ones_like(dom_logits_src)
        dom_tgt_labels = torch.zeros_like(dom_logits_tgt)

        domain_loss = (
            domain_criterion(dom_logits_src, dom_src_labels) +
            domain_criterion(dom_logits_tgt, dom_tgt_labels)
        )

        loss = class_loss + domain_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log every 10 steps
        if global_step % 10 == 0:
            wandb.log({
                "step": global_step,
                "epoch": epoch,
                "loss": loss.item(),
                "class_loss": class_loss.item(),
                "domain_loss": domain_loss.item(),
                "lambda": lambda_,
            })

        global_step += 1

    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

# Calculate accuracies after training
print("\n" + "="*50)
print("Calculating accuracies after DANN adaptation...")
print("="*50)

real_test_acc = evaluate_accuracy(clip_model, dann, real_test_loader)
infograph_test_acc = evaluate_accuracy(clip_model, dann, infograph_eval_loader)

print(f"Real Test Accuracy: {real_test_acc:.4f}")
print(f"Infograph Test Accuracy: {infograph_test_acc:.4f}")

wandb.log({
    "final/real_test_accuracy": real_test_acc,
    "final/infograph_test_accuracy": infograph_test_acc,
})

#11. t-sne after domain adaption

print("\nExtracting features AFTER adaptation...")

src_feats, src_dom = extract_tsne_features(
    clip_model, dann, source_loader, domain_label=1
)
tgt_feats, tgt_dom = extract_tsne_features(
    clip_model, dann, target_loader, domain_label=0
)

X = np.vstack([src_feats, tgt_feats])
y = np.hstack([src_dom, tgt_dom])

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X)

plt.figure(figsize=(6, 6))
plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], label="Source", alpha=0.6)
plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], label="Target", alpha=0.6)
plt.legend()
plt.title("t-SNE AFTER Domain Adaptation")
plt.savefig("tsne_after_adaptation.png", dpi=150, bbox_inches='tight')
wandb.log({"tsne_after_adaptation": wandb.Image("tsne_after_adaptation.png")})
plt.show()

# Finish wandb run
wandb.finish()





