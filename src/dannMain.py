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
from torch.utils.data import random_split
import open_clip
import torch.optim as optim

METHOD = "unfrozen" # "unfrozen" # "frozen"
TEST_ON = "infograph" # "clipart" # sketch
epochs = 20
SCHEDULER = "stepLR" # stepLR
LR_clip = "1e-4"
LR_dannHead = "1e-2"
# NORMALIZE = "yes"
# wandb initialize
if METHOD=="frozen":
    wandb.init(project="applied-dl-domain-adaptation", name=f"real_{TEST_ON}_clip{METHOD}_Dann_adaptedLamda_{LR_clip}clip_{LR_dannHead}dann_{SCHEDULER}")
elif METHOD=="unfrozen":
    wandb.init(project="applied-dl-domain-adaptation", name=f"real_{TEST_ON}_clip{METHOD}_Dann_adaptedLamda_{LR_clip}clip_{LR_dannHead}dann_{SCHEDULER}")

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
# class DANNHead(nn.Module):
#     def __init__(self, feature_dim, num_classes, hidden_dim=512):
#         super().__init__()
#         # 1 linear
#         self.adapter = nn.Sequential(
#             nn.Linear(feature_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.2)
#         )
#         # 1 linear
#         self.class_classifier = nn.Linear(hidden_dim, num_classes)

#         # 2 linear
#         self.domain_classifier = nn.Sequential(
#             nn.Linear(hidden_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )

#     def forward(self, features, lambda_=0.0):
#         features = self.adapter(features)

#         class_logits = self.class_classifier(features)

#         rev_features = grl(features, lambda_)
#         domain_logits = self.domain_classifier(rev_features)

#         return class_logits, domain_logits

#     def extract_features(self, features):
#         return self.adapter(features)

class DANNHead(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden_dim=1024):
        super().__init__()

        # 3-layer adapter
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 3-layer class classifier
        self.class_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, num_classes)
        )

        # 3-layer domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
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
clip_model.eval()
# Load fine-tuned weights
#print("Loading fine-tuned weights...")
# checkpoint = torch.load("clip_real_finetuned.pth", map_location=device)
# clip_model.visual.load_state_dict(checkpoint["clip_visual_state_dict"])
# clip_model.eval()
# print("Model loaded.")

######################
# option a: freezing clip 
if METHOD=="frozen":
    for p in clip_model.parameters():
        p.requires_grad = False
elif METHOD=="unfrozen":
    for p in clip_model.parameters():
        p.requires_grad = True

# option b: not freezing clip -  no change
######################

# source_loader: (image, label) - labeled source domain
# target_loader: (image) - unlabeled target domain for adaptation

# Load Datasets
print("Loading Datasets...")

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
real_val_dataset, real_test_dataset = random_split(real_testing_dataset, [val_size, test_size], generator=torch.Generator().manual_seed(42))


# Target domain: UNLABELED for adaptation
if TEST_ON =="infograph":
    datapath_name = "infograph"
elif  TEST_ON =="clipart":
    datapath_name = "clipart"
elif  TEST_ON =="sketch":
    datapath_name = "sketch"

target_train_dataset = UnlabeledDomainDataset(
    root_dir=f"data/{datapath_name}",
    txt_file=f"data/{datapath_name}/{datapath_name}_train.txt",
    transform=preprocess,
)

# Also load labeled infograph TEST dataset for evaluation (held-out)
target_test_dataset = DomainNetListDataset(
    root_dir=f"data/{datapath_name}",
    txt_file=f"data/{datapath_name}/{datapath_name}_test.txt",
    transform=preprocess,
)
val_size = int(len(target_test_dataset) * 0.5)
test_size = len(target_test_dataset) - val_size
target_val_dataset, target_test_dataset = random_split(target_test_dataset, [val_size, test_size], generator=torch.Generator().manual_seed(42))


num_classes = max(label for _, label in real_train_dataset.samples) + 1
print(f"Loaded {len(real_train_dataset)} training samples, {num_classes} classes")
print("Datasets loaded.")

print("Preparing DataLoaders...")
real_train_loader = DataLoader(real_train_dataset, batch_size=64, shuffle=True, num_workers=4) # train data (S)
real_test_loader = DataLoader(real_test_dataset, batch_size=64, shuffle=False, num_workers=4) # test  (S)
real_val_loader = DataLoader(real_val_dataset, batch_size=64, shuffle=False, num_workers=4) # val (S)
target_train_loader = DataLoader(target_train_dataset, batch_size=64, shuffle=True, num_workers=4) # train data
target_test_loader = DataLoader(target_test_dataset, batch_size=64, shuffle=False, num_workers=4)
target_val_loader = DataLoader(target_val_dataset, batch_size=64, shuffle=False, num_workers=4)
print("DataLoaders ready.")

#7. Initialize DANN

print(f"Initializing DANN with {num_classes} classes...")
feature_dim = clip_model.visual.output_dim

dann = DANNHead(feature_dim, num_classes).to(device)

# optimizer
if METHOD=="frozen":
        optimizer = torch.optim.Adam( dann.parameters(), lr=1e-3, weight_decay=1e-5)
elif METHOD=="unfrozen":
    optimizer = torch.optim.Adam([
        {"params": clip_model.parameters(), "lr": LR_clip}, # clip backbone # baseline at 1e-6
        {"params": dann.parameters(), "lr": LR_dannHead} # dann head # baseline at 1e-3
    ], weight_decay=1e-5)


## option b: not freezing CLIP
# optimizer = torch.optim.Adam([
#     {"params": clip_model.parameters(), "lr": 1e-6},
#     {"params": dann.parameters(), "lr": 1e-3}
# ], weight_decay=1e-5)
######################
if SCHEDULER =="stepLR":
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5,gamma=0.1)
elif SCHEDULER =="cosine":
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs)
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
        clip_feats = F.normalize(clip_feats, dim=-1) # added normalization
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
    clip_model, dann, real_train_loader, domain_label=1
)
tgt_feats, tgt_dom = extract_tsne_features(
    clip_model, dann, target_train_loader, domain_label=0
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
total_steps = epochs * min(len(real_train_loader), len(target_train_loader))
global_step = 0

for epoch in range(epochs):
    dann.train()
    
    for (x_src, y_src), x_tgt in zip(real_train_loader, target_train_loader):
        x_src = x_src.to(device)
        y_src = y_src.to(device)
        x_tgt = x_tgt.to(device)

        lambda_ = dann_lambda(global_step, total_steps)

        ######################
        # # option a: freezing clip 
        # with torch.no_grad():
        #     f_src = clip_model.encode_image(x_src).float()
        #     f_tgt = clip_model.encode_image(x_tgt).float()

        # option b: not freezing clip 
        # f_src = clip_model.encode_image(x_src).float()
        # f_tgt = clip_model.encode_image(x_tgt).float()
        if METHOD =="frozen":
            with torch.no_grad():
                f_src = F.normalize(clip_model.encode_image(x_src).float(), dim=-1) # adding normalization 
                f_tgt = F.normalize(clip_model.encode_image(x_tgt).float(), dim=-1) # adding normalization
        elif METHOD =="unfrozen":
            f_src = F.normalize(clip_model.encode_image(x_src).float(), dim=-1) # adding normalization 
            f_tgt = F.normalize(clip_model.encode_image(x_tgt).float(), dim=-1) # adding normalization
        ######################

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
    
    # eval 
    real_val_acc = evaluate_accuracy(clip_model, dann, real_val_loader)
    target_val_acc = evaluate_accuracy(clip_model, dann, target_val_loader)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f} | real_val_acc: {real_val_acc} | target_val_acc: {target_val_acc}")

    wandb.log({
        "epoch": epoch,
        "real_val_acc": real_val_acc,
        "target_val_acc": target_val_acc,
    })

    scheduler.step()

# Calculate accuracies after training
print("\n" + "="*50)
print("Calculating accuracies after DANN adaptation...")
print("="*50)

real_test_acc = evaluate_accuracy(clip_model, dann, real_test_loader)
target_test_acc = evaluate_accuracy(clip_model, dann, target_test_loader)

print(f"Real Test Accuracy: {real_test_acc:.4f}")
print(f"Target Test Accuracy: {target_test_acc:.4f}")

wandb.log({
    "final/real_test_accuracy": real_test_acc,
    "final/real_val_accuracy": real_val_acc,
    "final/target_test_accuracy": target_test_acc,
    "final/target_val_accuracy": target_val_acc
})

wandb.finish()



# save model 
if METHOD == "frozen":
    model_save_path = f"models/dann/real_{TEST_ON}_clip{METHOD}_Dann_adaptedLambda.pth"
elif METHOD =="unfrozen":
    model_save_path = f"models/dann/real_{TEST_ON}_clip{METHOD}_Dann_adaptedLambda.pth"



torch.save({
    "clip_visual_state_dict": clip_model.visual.state_dict(),
    "dann_state_dict": dann.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
}, model_save_path)




## with and without clip freezing (2 cases)
## constant and adaptive lamda values (2 cases)
## SGD
## tsne - before after


#### after all things are done: ABLATIONS
# a) Table showing (zero  shot -> finetuning -> dann) (for each real, inforgraph, clipart)
# b) add mlp finetuning + domain adaptation
# c) domain generlllization gap 