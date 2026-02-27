import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.manifold import TSNE
import open_clip

# -------------------------
# Config
# -------------------------
source_dir = "data/real"
target_dir = "data/clipart"   # change if needed
ckpt_path  = "models/dann/Dann_adaptedLambda.pth"
batch_size = 256
max_images_per_domain = 2000
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# DANNHead (must match training)
# -------------------------
class DANNHead(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, features, lambda_=1.0):
        z = self.adapter(features)
        class_logits = self.class_classifier(z)
        domain_logits = self.domain_classifier(z)  # GRL not needed for inference/tsne
        return class_logits, domain_logits

    def extract_features(self, features):
        return self.adapter(features)

# -------------------------
# Dataset (images only)
# -------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_images=None):
        self.transform = transform
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(root, f))
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

# -------------------------
# 1) Build CLIP + preprocess
# -------------------------
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
clip_model = clip_model.to(device)

# -------------------------
# 2) Load checkpoint
# -------------------------
ckpt = torch.load(ckpt_path, map_location=device)

# Load visual weights (in frozen mode this equals base weights, but ok)
clip_model.visual.load_state_dict(ckpt["clip_visual_state_dict"])

# Infer num_classes from saved DANN state_dict (no guessing)
dann_state = ckpt["dann_state_dict"]
num_classes = dann_state["class_classifier.2.weight"].shape[0]

dann = DANNHead(feature_dim=clip_model.visual.output_dim, num_classes=num_classes).to(device)
dann.load_state_dict(dann_state)

clip_model.eval()
dann.eval()

# -------------------------
# 3) Load data
# -------------------------
src_ds = ImageFolderDataset(source_dir, transform=preprocess, max_images=max_images_per_domain)
tgt_ds = ImageFolderDataset(target_dir, transform=preprocess, max_images=max_images_per_domain)
src_loader = DataLoader(src_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
tgt_loader = DataLoader(tgt_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# -------------------------
# 4) Extract ADAPTED features for t-SNE
# -------------------------
@torch.no_grad()
def extract_adapted(loader, domain_label):
    feats_list, dom_list = [], []
    for images in loader:
        images = images.to(device, non_blocking=True)

        clip_feats = clip_model.encode_image(images).float()
        clip_feats = F.normalize(clip_feats, dim=-1)          # match your training/eval
        adapted = dann.extract_features(clip_feats)           # (B, 256)

        feats_list.append(adapted.cpu().numpy())
        dom_list.append(np.full(len(images), domain_label))
    return np.vstack(feats_list), np.hstack(dom_list)

print("Extracting adapted features...")
src_feats, src_dom = extract_adapted(src_loader, domain_label=1)
tgt_feats, tgt_dom = extract_adapted(tgt_loader, domain_label=0)

X = np.vstack([src_feats, tgt_feats])
y = np.hstack([src_dom, tgt_dom])

# -------------------------
# 5) t-SNE + plot
# -------------------------
print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
X_2d = tsne.fit_transform(X)

plt.figure(figsize=(6, 6))
plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], label="Source (real)", alpha=0.6, s=8)
plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], label="Target (clipart)", alpha=0.6, s=8)
plt.legend()
plt.title("t-SNE of DANN-adapted features (adapter output)")
plt.savefig("tsne_dann_adapted_features_feb27_old.png", dpi=150, bbox_inches="tight")
plt.show()

print("Saved: tsne_dann_adapted_features.png")