import os
import math
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset, random_split

import torch.optim as optim
import open_clip
import wandb
from PIL import Image


# ======================
# Config
# ======================
TEST_ON = "clipart"          # "clipart" | "sketch" | "infograph" | "painting"
SCHEDULER = "cosine"         # "cosine" | "stepLR"

EPOCHS_STAGE1 = 6            # head warmup (no domain loss, CLIP visual frozen)
EPOCHS_STAGE2 = 14           # adversarial (domain loss + GRL, CLIP visual unfrozen)
epochs = EPOCHS_STAGE1 + EPOCHS_STAGE2

BATCH_SIZE = 512
NUM_WORKERS = 4
WEIGHT_DECAY = 1e-4

LR_head_stage1 = 1e-3        # train adapter+class+domain heads in stage1 (domain unused)
LR_visual_stage2 = 1e-5      # unfreeze CLIP visual in stage2
LR_head_stage2 = 1e-3        # keep head LR in stage2 (often fine)

LAM_MAX_STAGE2 = 0.6       # IMPORTANT: start with 0.1 for DomainNet; try 0.3 later
MODEL_SAVE_PATH = "models/dann/Dann_visualOnly_twoStage.pth"


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
    def __init__(self, feature_dim, num_classes, hidden_dim=512):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
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
        feats = self.adapter(features)
        class_logits = self.class_classifier(feats)

        rev_feats = grl(feats, lambda_)
        domain_logits = self.domain_classifier(rev_feats)
        return class_logits, domain_logits


def dann_lambda(step, max_steps, lam_max=1.0):
    p = step / max_steps
    return lam_max * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0)


# ======================
# Datasets
# ======================
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


# ======================
# Eval
# ======================
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


# ======================
# Helpers: param freezing
# ======================
def freeze_all_clip(clip_model):
    for p in clip_model.parameters():
        p.requires_grad = False


def unfreeze_visual_only(clip_model):
    for p in clip_model.visual.parameters():
        p.requires_grad = True


# ======================
# Main
# ======================
def main():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    wandb.init(
        project="applied-dl-domain-adaptation",
        name=f"real_{TEST_ON}_twoStage_visualOnly_stage1{EPOCHS_STAGE1}_stage2{EPOCHS_STAGE2}_lam{LAM_MAX_STAGE2}",
        config={
            "TEST_ON": TEST_ON,
            "epochs_total": epochs,
            "EPOCHS_STAGE1": EPOCHS_STAGE1,
            "EPOCHS_STAGE2": EPOCHS_STAGE2,
            "BATCH_SIZE": BATCH_SIZE,
            "LR_head_stage1": LR_head_stage1,
            "LR_visual_stage2": LR_visual_stage2,
            "LR_head_stage2": LR_head_stage2,
            "LAM_MAX_STAGE2": LAM_MAX_STAGE2,
            "SCHEDULER": SCHEDULER,
        },
    )

    # ---- Load CLIP ----
    print("Loading CLIP base model...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model = clip_model.to(device)
    print("CLIP model loaded.")

    # ---- Data ----
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
        generator=torch.Generator().manual_seed(42),
    )

    target_train_dataset = UnlabeledDomainDataset(
        root_dir=f"data/{TEST_ON}",
        txt_file=f"data/{TEST_ON}/{TEST_ON}_train.txt",
        transform=preprocess,
    )

    target_test_dataset = DomainNetListDataset(
        root_dir=f"data/{TEST_ON}",
        txt_file=f"data/{TEST_ON}/{TEST_ON}_test.txt",
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

    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    # ==========================
    # STAGE 1: Head warmup only
    # ==========================
    print("\n" + "=" * 60)
    print(f"STAGE 1: Warmup head only for {EPOCHS_STAGE1} epochs (no domain loss)")
    print("=" * 60)

    freeze_all_clip(clip_model)  # visual frozen too
    clip_model.eval()            # keep CLIP frozen behavior stable
    dann.train()

    optimizer = torch.optim.Adam(
        [{"params": dann.parameters(), "lr": LR_head_stage1}],
        weight_decay=WEIGHT_DECAY,
    )

    # optional scheduler for stage1
    if SCHEDULER == "stepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    elif SCHEDULER == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_STAGE1)
    else:
        raise ValueError(f"Unknown scheduler: {SCHEDULER}")

    global_step = 0

    for epoch in range(EPOCHS_STAGE1):
        for x_src, y_src in real_train_loader:
            x_src = x_src.to(device, non_blocking=True)
            y_src = y_src.to(device, non_blocking=True)

            with torch.no_grad():  # CLIP frozen in stage1
                f_src = clip_model.encode_image(x_src).float()
                f_src = F.normalize(f_src, dim=-1)

            class_logits, _ = dann(f_src, lambda_=0.0)  # lambda not used in stage1
            class_loss = class_criterion(class_logits, y_src)

            optimizer.zero_grad(set_to_none=True)
            class_loss.backward()
            optimizer.step()

            if global_step % 50 == 0:
                wandb.log(
                    {
                        "stage": 1,
                        "step": global_step,
                        "epoch": epoch,
                        "class_loss": class_loss.item(),
                        "lr_head": optimizer.param_groups[0]["lr"],
                    }
                )

            global_step += 1

        real_val_acc = evaluate_accuracy(clip_model, dann, real_val_loader, device)
        target_val_acc = evaluate_accuracy(clip_model, dann, target_val_loader, device)
        print(
            f"[Stage1] Epoch {epoch+1}/{EPOCHS_STAGE1} | real_val_acc={real_val_acc:.4f} | target_val_acc={target_val_acc:.4f}"
        )
        wandb.log(
            {
                "stage": 1,
                "epoch": epoch,
                "real_val_acc": real_val_acc,
                "target_val_acc": target_val_acc,
            }
        )
        scheduler.step()

    


    # print("\n" + "=" * 60)
    # print(f"STAGE 2 ABLATION A: Unfreeze visual, source-only fine-tune (NO domain loss)")
    # print("=" * 60)

    # # Freeze all CLIP then unfreeze ONLY visual
    # freeze_all_clip(clip_model)
    # unfreeze_visual_only(clip_model)
    # clip_model.train()
    # dann.train()

    # optimizer = torch.optim.Adam(
    #     [
    #         {"params": clip_model.visual.parameters(), "lr": LR_visual_stage2},
    #         {"params": dann.parameters(), "lr": LR_head_stage2},
    #     ],
    #     weight_decay=WEIGHT_DECAY,
    # )

    # if SCHEDULER == "stepLR":
    #     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # elif SCHEDULER == "cosine":
    #     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_STAGE2)
    # else:
    #     raise ValueError(f"Unknown scheduler: {SCHEDULER}")

    # for epoch2 in range(EPOCHS_STAGE2):
    #     for x_src, y_src in real_train_loader:
    #         x_src = x_src.to(device, non_blocking=True)
    #         y_src = y_src.to(device, non_blocking=True)

    #         # Unfrozen visual => gradients flow
    #         f_src = clip_model.encode_image(x_src).float()
    #         f_src = F.normalize(f_src, dim=-1)

    #         # lambda_ doesn't matter since we won't use domain branch gradients
    #         class_logits, _ = dann(f_src, lambda_=0.0)

    #         class_loss = class_criterion(class_logits, y_src)
    #         loss = class_loss  # <- NO domain loss

    #         optimizer.zero_grad(set_to_none=True)
    #         loss.backward()
    #         optimizer.step()

    #     real_val_acc = evaluate_accuracy(clip_model, dann, real_val_loader, device)
    #     target_val_acc = evaluate_accuracy(clip_model, dann, target_val_loader, device)

    #     print(
    #         f"[AblA] Epoch {epoch2+1}/{EPOCHS_STAGE2} | real_val_acc={real_val_acc:.4f} | target_val_acc={target_val_acc:.4f}"
    #     )

    #     wandb.log(
    #         {
    #             "ablation": "A_unfreeze_no_domain",
    #             "epoch2": epoch2,
    #             "real_val_acc": real_val_acc,
    #             "target_val_acc": target_val_acc,
    #             "loss": loss.item(),
    #         }
    #     )

    #     scheduler.step()

    # ==========================================
    # STAGE 2: Adversarial + unfreeze visual only
    # ==========================================
    print("\n" + "=" * 60)
    print(f"STAGE 2: Adversarial training for {EPOCHS_STAGE2} epochs (GRL + domain loss)")
    print("=" * 60)

    freeze_all_clip(clip_model)
    unfreeze_visual_only(clip_model)
    clip_model.train()
    dann.train()

    optimizer = torch.optim.Adam(
        [
            {"params": clip_model.visual.parameters(), "lr": LR_visual_stage2},
            {"params": dann.parameters(), "lr": LR_head_stage2},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    if SCHEDULER == "stepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    elif SCHEDULER == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_STAGE2)
    else:
        raise ValueError(f"Unknown scheduler: {SCHEDULER}")

    total_steps_stage2 = EPOCHS_STAGE2 * len(real_train_loader)
    stage2_step = 0

    for epoch2 in range(EPOCHS_STAGE2):
        tgt_iter = cycle(target_train_loader)

        for x_src, y_src in real_train_loader:
            x_tgt = next(tgt_iter)

            x_src = x_src.to(device, non_blocking=True)
            y_src = y_src.to(device, non_blocking=True)
            x_tgt = x_tgt.to(device, non_blocking=True)

            lambda_ = dann_lambda(stage2_step, total_steps_stage2, lam_max=LAM_MAX_STAGE2)

            # IMPORTANT: stage2 has gradients into clip_model.visual (no no_grad)
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

            if stage2_step % 10 == 0:
                wandb.log(
                    {
                        "stage": 2,
                        "step": stage2_step,
                        "epoch2": epoch2,
                        "loss": loss.item(),
                        "class_loss": class_loss.item(),
                        "domain_loss": domain_loss.item(),
                        "domain_accuracy": dom_acc.item(),
                        "lambda": lambda_,
                        "lr_visual": optimizer.param_groups[0]["lr"],
                        "lr_head": optimizer.param_groups[1]["lr"],
                    }
                )

            stage2_step += 1

        real_val_acc = evaluate_accuracy(clip_model, dann, real_val_loader, device)
        target_val_acc = evaluate_accuracy(clip_model, dann, target_val_loader, device)
        print(
            f"[Stage2] Epoch {epoch2+1}/{EPOCHS_STAGE2} | real_val_acc={real_val_acc:.4f} | target_val_acc={target_val_acc:.4f}"
        )
        wandb.log(
            {
                "stage": 2,
                "epoch2": epoch2,
                "real_val_acc": real_val_acc,
                "target_val_acc": target_val_acc,
            }
        )

        scheduler.step()

    # ---- final test ----
    print("\n" + "=" * 50)
    print("Final test accuracies")
    print("=" * 50)

    real_test_acc = evaluate_accuracy(clip_model, dann, real_test_loader, device)
    target_test_acc = evaluate_accuracy(clip_model, dann, target_test_loader, device)
    print(f"Real Test Accuracy:   {real_test_acc:.4f}")
    print(f"Target Test Accuracy: {target_test_acc:.4f}")

    wandb.log(
        {
            "final/real_test_accuracy": real_test_acc,
            "final/target_test_accuracy": target_test_acc,
        }
    )

    torch.save(
        {
            "clip_visual_state_dict": clip_model.visual.state_dict(),
            "dann_state_dict": dann.state_dict(),
            "epochs_stage1": EPOCHS_STAGE1,
            "epochs_stage2": EPOCHS_STAGE2,
            "TEST_ON": TEST_ON,
            "LR_visual_stage2": LR_visual_stage2,
            "LR_head_stage1": LR_head_stage1,
            "LR_head_stage2": LR_head_stage2,
            "LAM_MAX_STAGE2": LAM_MAX_STAGE2,
        },
        MODEL_SAVE_PATH,
    )
    print(f"Saved model to: {MODEL_SAVE_PATH}")

    wandb.finish()


if __name__ == "__main__":
    main()