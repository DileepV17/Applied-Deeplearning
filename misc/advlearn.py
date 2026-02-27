import os
import math
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset, random_split

import torch.optim as optim
from itertools import cycle

import open_clip
import wandb


# ======================
# Config
# ======================
TEST_ON = "real"  # "clipart" | "sketch" | "infograph" | "painting"
SCHEDULER = "cosine"  # "cosine" | "stepLR"
epochs = 10

LR_visual = 1e-6      # CLIP visual encoder LR (small)
LR_dann = 1e-3      # DANN head LR
BATCH_SIZE = 512

NUM_WORKERS = 8
WEIGHT_DECAY = 1e-5
LAM_MAX = 1.0
MODEL_SAVE_PATH = "models/dann/Dann_visualOnly_adaptedLambda.pth"
print("using source as clipart and target as Real")


# ======================
# GRL + DANN Head
# ======================
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


class DANNHead(nn.Module):
    """Shared adapter + class classifier + domain discriminator (with GRL)"""
    def __init__(self, feature_dim, num_classes, hidden_dim=512):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Dropout(0.1),
            nn.ReLU(),
        )

        self.class_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, features, lambda_=1.0):
        feats = self.adapter(features)

        class_logits = self.class_classifier(feats)

        rev_feats = grl(feats, lambda_)
        domain_logits = self.domain_classifier(rev_feats)

        return class_logits, domain_logits

    def extract_features(self, features):
        return self.adapter(features)


def dann_lambda(step, max_steps, lam_max=1.0):
    p = step / max_steps
    return lam_max * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0)

#dataset classes
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
    """Returns only images (no labels) for unsupervised domain adaptation"""
    def __init__(self, root_dir, txt_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        with open(txt_file, "r") as f:
            for line in f:
                path, _ = line.strip().split()  # ignore label
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


# defining evaluation function
@torch.no_grad()
def evaluate_accuracy(clip_model, dann, loader, device):
    clip_model.eval()
    dann.eval()
    correct, total = 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        clip_feats = clip_model.encode_image(images).float()
        clip_feats = F.normalize(clip_feats, dim=-1)

        class_logits, _ = dann(clip_feats, lambda_=1.0)
        preds = class_logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / max(total, 1)


# wandb initialize
def main():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    wandb.init(
        project="applied-dl-domain-adaptation",
        name=f"real_{TEST_ON}_visualOnly_DANN_lamMax{LAM_MAX}_lrV{LR_visual}_lrD{LR_dann}_{SCHEDULER}",
        config={
            "TEST_ON": TEST_ON,
            "epochs": epochs,
            "BATCH_SIZE": BATCH_SIZE,
            "LR_visual": LR_visual,
            "LR_dann": LR_dann,
            "SCHEDULER": SCHEDULER,
            "LAM_MAX": LAM_MAX,
        },
    )

    # ---- Load CLIP ----
    print("Loading CLIP base model...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
    )
    clip_model = clip_model.to(device)
    print("CLIP model loaded.")

    # ---- Train ONLY visual encoder ----
    for p in clip_model.parameters():
        p.requires_grad = False
    for p in clip_model.visual.parameters():
        p.requires_grad = True
    clip_model.train()

    # ---- Datasets ----
    print("Loading datasets...")

    # Source = real (labeled)
    real_train_dataset = DomainNetListDataset(
        root_dir="data/clipart",
        txt_file="data/clipart/clipart_train.txt",
        transform=preprocess,
    )

    real_testing_dataset = DomainNetListDataset(
        root_dir="data/clipart",
        txt_file="data/clipart/clipart_test.txt",
        transform=preprocess,
    )
    val_size = int(len(real_testing_dataset) * 0.5)
    test_size = len(real_testing_dataset) - val_size
    real_val_dataset, real_test_dataset = random_split(
        real_testing_dataset,
        [val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Target (unlabeled train + labeled test/val for eval)
    datapath_name = TEST_ON
    target_train_dataset = UnlabeledDomainDataset(
        root_dir=f"data/{datapath_name}",
        txt_file=f"data/{datapath_name}/{datapath_name}_train.txt",
        transform=preprocess,
    )

    target_test_dataset = DomainNetListDataset(
        root_dir=f"data/{datapath_name}",
        txt_file=f"data/{datapath_name}/{datapath_name}_test.txt",
        transform=preprocess,
    )
    val_size = int(len(target_test_dataset) * 0.5)
    test_size = len(target_test_dataset) - val_size
    target_val_dataset, target_test_dataset = random_split(
        target_test_dataset,
        [val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    num_classes = max(label for _, label in real_train_dataset.samples) + 1
    print(f"Loaded {len(real_train_dataset)} source train samples, num_classes={num_classes}")

    # ---- Loaders ----
    print("Preparing DataLoaders...")
    real_train_loader = DataLoader(
        real_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    real_val_loader = DataLoader(
        real_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    real_test_loader = DataLoader(
        real_test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    target_train_loader = DataLoader(
        target_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    target_val_loader = DataLoader(
        target_val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    target_test_loader = DataLoader(
        target_test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    print("DataLoaders ready.")

    # ---- DANN head ----
    feature_dim = clip_model.visual.output_dim
    dann = DANNHead(feature_dim, num_classes).to(device)

    # defining Optimizer & Scheduler
    optimizer = torch.optim.Adam(
        [
            {"params": clip_model.visual.parameters(), "lr": LR_visual},
            {"params": dann.parameters(), "lr": LR_dann},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    if SCHEDULER == "stepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    elif SCHEDULER == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        raise ValueError(f"Unknown scheduler: {SCHEDULER}")

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    
    # Training loop 
    
    print("Starting DANN training (visual-only CLIP encoder)...")

    total_steps = epochs * len(real_train_loader)
    global_step = 0

    for epoch in range(epochs):
        clip_model.train()
        dann.train()

        tgt_iter = cycle(target_train_loader)

        for x_src, y_src in real_train_loader:
            x_tgt = next(tgt_iter)

            x_src = x_src.to(device, non_blocking=True)
            y_src = y_src.to(device, non_blocking=True)
            x_tgt = x_tgt.to(device, non_blocking=True)

            lambda_ = dann_lambda(global_step, total_steps, lam_max=LAM_MAX)

            # IMPORTANT: no torch.no_grad(), no detach()
            f_src = clip_model.encode_image(x_src).float()
            f_tgt = clip_model.encode_image(x_tgt).float()

            f_src = F.normalize(f_src, dim=-1)
            f_tgt = F.normalize(f_tgt, dim=-1)

            class_logits, dom_logits_src = dann(f_src, lambda_)
            _, dom_logits_tgt = dann(f_tgt, lambda_)

            class_loss = class_criterion(class_logits, y_src)

            dom_src_labels = torch.ones_like(dom_logits_src)
            dom_tgt_labels = torch.zeros_like(dom_logits_tgt)

            domain_loss = (
                domain_criterion(dom_logits_src, dom_src_labels)
                + domain_criterion(dom_logits_tgt, dom_tgt_labels)
            )

            loss = class_loss + domain_loss  # Option A

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                dom_preds_src = (torch.sigmoid(dom_logits_src) > 0.5).float()
                dom_preds_tgt = (torch.sigmoid(dom_logits_tgt) > 0.5).float()
                dom_acc = (
                    dom_preds_src.eq(dom_src_labels).float().mean()
                    + dom_preds_tgt.eq(dom_tgt_labels).float().mean()
                ) / 2.0

            if global_step % 10 == 0:
                wandb.log(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": loss.item(),
                        "class_loss": class_loss.item(),
                        "domain_loss": domain_loss.item(),
                        "domain_accuracy": dom_acc.item(),
                        "lambda": lambda_,
                        "lr_visual": optimizer.param_groups[0]["lr"],
                        "lr_dann": optimizer.param_groups[1]["lr"],
                    }
                )

            global_step += 1

        # ---- epoch eval ----
        real_val_acc = evaluate_accuracy(clip_model, dann, real_val_loader, device)
        target_val_acc = evaluate_accuracy(clip_model, dann, target_val_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"loss={loss.item():.4f} | real_val_acc={real_val_acc:.4f} | target_val_acc={target_val_acc:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch,
                "real_val_acc": real_val_acc,
                "target_val_acc": target_val_acc,
            }
        )

        scheduler.step()

    # ---- final test ----
    print("\n" + "=" * 50)
    print("Calculating accuracies after DANN adaptation.")
    print("=" * 50)

    real_test_acc = evaluate_accuracy(clip_model, dann, real_test_loader, device)
    target_test_acc = evaluate_accuracy(clip_model, dann, target_test_loader, device)

    print(f"Real Test Accuracy:   {real_test_acc:.4f}")
    print(f"Target Test Accuracy: {target_test_acc:.4f}")

    wandb.log(
        {
            "final/real_test_accuracy": real_test_acc,
            "final/target_test_accuracy": target_test_acc,
            "final/real_val_accuracy": real_val_acc,
            "final/target_val_accuracy": target_val_acc,
        }
    )

    # ---- save ----
    torch.save(
        {
            "clip_visual_state_dict": clip_model.visual.state_dict(),
            "dann_state_dict": dann.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epochs": epochs,
            "TEST_ON": TEST_ON,
            "LR_visual": LR_visual,
            "LR_dann": LR_dann,
            "LAM_MAX": LAM_MAX,
        },
        MODEL_SAVE_PATH,
    )
    print(f"Saved model to: {MODEL_SAVE_PATH}")

    wandb.finish()


if __name__ == "__main__":
    main()