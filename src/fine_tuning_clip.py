import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import open_clip
from tqdm import tqdm
from PIL import Image
import wandb
from torch.utils.data import Subset


# wandb initialize
wandb.init(
    project="applied-dl-domain-adaptation",
    name="finetune_clip_realImages",
)

device = "cuda" if torch.cuda.is_available() else "cpu" # setting up for gpu

print("STEP 1: Loading CLIP...")
model, _, preprocess = open_clip.create_model_and_transforms( 'ViT-B-32', pretrained='openai' )
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model = model.to(device)
print("STEP 1 COMPLETE")


# for train loader 
class DomainNetListDataset(torch.utils.data.Dataset):
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
        img = Image.open(f"{self.root_dir}/{rel_path}").convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


print("STEP 2: Loading the dataset...")
train_transform = preprocess
test_transform = preprocess
train_dataset = DomainNetListDataset(
    root_dir="data/real/",
    txt_file="data/real/real_train.txt",
    transform=train_transform
)
real_test_dataset = DomainNetListDataset(
    root_dir="data/real/",
    txt_file="data/real/real_test.txt",
    transform=test_transform
)

infograph_test_dataset = DomainNetListDataset(
    root_dir="data/infograph",
    txt_file="data/infograph/infograph_test.txt",
    transform=test_transform
)
num_classes = max([label for _, label in train_dataset.samples]) + 1   # to check


########
# slicing dataset below
# disable this for full runs 
# train_dataset = Subset(train_dataset, range(5000))
# real_test_dataset = Subset(real_test_dataset, range(500))
# infograph_test_dataset = Subset(infograph_test_dataset, range(500))
########

print("STEP 2 COMPLETE")


print("STEP 3: Creating the data loaders...")
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
real_test_loader = DataLoader(real_test_dataset, batch_size=64, shuffle=False, num_workers=0)
infograph_test_loader = DataLoader(infograph_test_dataset, batch_size=64, shuffle=False, num_workers=0)
print("STEP 3 COMPLETE")

# num_classes = len(train_dataset.classes) # to check



image_features_dim = model.visual.output_dim
classifier = nn.Linear(image_features_dim, num_classes).to(device) # linear layer

def train_one_epoch():
    model.train()
    classifier.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in tqdm(enumerate(train_loader)):
        images, labels = images.to(device), labels.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam( list(model.visual.parameters()) + list(classifier.parameters()), lr=1e-5 )
        optimizer.zero_grad() 

        image_features = model.encode_image(images) # we only use the clip image encoder and ignore the text encoder since we have only images and no text pairs

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # to check

        logits = classifier(image_features)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        # LOG EVERY 10 STEPS - you can change this 10 to any number you like
        if (batch_idx + 1) % 10 == 0:
            wandb.log({
                "train/loss_step": loss.item(),
                "train/acc_step": correct / total,
                "train/step": batch_idx + 1
            })
            print(
                f"[EPOCH {epoch}]:"
                f"Train Step [{batch_idx+1}/{len(train_loader)}] "
                f"Train Loss: {loss.item():.4f} "
                f"Train Acc: {correct/total:.4f}"
            )
    return total_loss / len(train_dataset), correct / total


def evaluate(loader):
    model.eval()
    classifier.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device) # sending images and labels to the gpu
            image_features = model.encode_image(images) # we call only the image encoder for our task 
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) # normalization - tocheck
            logits = classifier(image_features)

            _, preds = logits.max(1) 
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
            acc  = correct / total
    return acc


print("Training Begins...")
EPOCHS = 10
for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch()
    real_acc = evaluate(real_test_loader)
    infograph_acc = evaluate(infograph_test_loader)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Train Acc:  {train_acc:.4f}")
    print(f"  Source Test Acc (Real): {real_acc:.4f}")
    print(f"  Target Test Acc (Infograph): {infograph_acc:.4f}")

    wandb.log({
        "epoch": epoch + 1,
        "train/loss_epoch": train_loss,
        "train/acc_epoch": train_acc,
        "test/real_acc": real_acc,
        "test/infograph_acc": infograph_acc
    })
    

# todo: save model with .p file 
# todo: add f1 score (precision and recall)  - log in wandb - print


# todo: check the training
# todo: check if we should do normalization or no 
# todo: visualization - tsne 