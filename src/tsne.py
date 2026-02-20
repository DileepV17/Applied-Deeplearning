import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.manifold import TSNE
import open_clip

# =========================
# 1. Configuration
# =========================
source_dir = "data/real"      # <-- change this
target_dir = "data/infograph"    # <-- change this
batch_size = 256
max_images_per_domain = 2000  # limit for faster TSNE

device = "cuda" if torch.cuda.is_available() else "cpu"


# clip before domain adapt
print("Loading CLIP...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="openai"
)
clip_model = clip_model.to(device)
clip_model.eval()

# clip after domain adapt
for p in clip_model.parameters():
    p.requires_grad = False
for p in clip_model.visual.parameters():
    p.requires_grad = True


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_images=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.image_paths = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(root, f))
        
        if max_images:
            self.image_paths = self.image_paths[:max_images]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img



# =========================
# 4. Create Datasets & Loaders
# =========================
source_dataset = ImageFolderDataset(
    source_dir, transform=preprocess, max_images=max_images_per_domain
)

target_dataset = ImageFolderDataset(
    target_dir, transform=preprocess, max_images=max_images_per_domain
)

source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=False)
target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False)

# =========================
# 5. Feature Extraction
# =========================
@torch.no_grad()
def extract_features(loader, domain_label):
    features = []
    domains = []

    for images in loader:
        images = images.to(device)
        feats = clip_model.encode_image(images).float()
        features.append(feats.cpu().numpy())
        domains.append(np.full(len(images), domain_label))

    return np.vstack(features), np.hstack(domains)

print("Extracting features...")
src_feats, src_dom = extract_features(source_loader, domain_label=1)
tgt_feats, tgt_dom = extract_features(target_loader, domain_label=0)

X = np.vstack([src_feats, tgt_feats])
y = np.hstack([src_dom, tgt_dom])

# =========================
# 6. t-SNE
# =========================
print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X)

# =========================
# 7. Plot
# =========================
plt.figure(figsize=(6, 6))
plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], label="Source", alpha=0.6)
plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], label="Target", alpha=0.6)
plt.legend()
plt.title("t-SNE of CLIP Encoder Features")
plt.savefig("tsne_clip_only(real & infograph)-new.png", dpi=150, bbox_inches="tight")
plt.show()

print("Done.")
