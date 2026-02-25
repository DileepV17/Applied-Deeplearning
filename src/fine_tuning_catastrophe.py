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
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from collections import Counter

# wandb initialize
# run 1- wo LS 
# run2 = with LS
loss_type = "cross_entropy_woLS" # cross_entropy_withLS , weighted_crossEntropy, cross_entropy_woLS
scheduler = "cosine"
model_name = f"10epochs_infograph_clipart_2layers1024_0.001lr_run-{scheduler}-{loss_type}"
wandb.init(
    project="applied-dl-domain-adaptation",
    name=model_name,
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

    # openns the image based on the [ root dir + rel path (inside the txt) ]
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

clipart_test_dataset = DomainNetListDataset(
    root_dir="data/clipart",
    txt_file="data/clipart/clipart_test.txt",
    transform=preprocess,
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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
real_test_loader = DataLoader(real_test_dataset, batch_size=64, shuffle=False, num_workers=4)
infograph_test_loader = DataLoader(infograph_test_dataset, batch_size=64, shuffle=False, num_workers=4)
clipart_test_loader = DataLoader(clipart_test_dataset, batch_size=64, shuffle=False, num_workers=4)
print("STEP 3 COMPLETE")

# num_classes = len(train_dataset.classes) # to check



image_features_dim = model.visual.output_dim
## zero layer ##
# classifier = nn.Linear(image_features_dim, num_classes).to(device) # linear layer


# # - 1024 - 2 layers
classifier  = nn.Sequential(
    nn.Linear(image_features_dim, 1024),
    nn.ReLU(inplace=True),
    nn.Dropout(0.1), #droupout changed to 0.1       
    nn.Linear(1024, num_classes)
).to(device)

# ## 5 layers
# classifier  = nn.Sequential(
#     nn.Linear(image_features_dim, 512),
#     nn.ReLU(inplace=True),
#     nn.Dropout(0.1), #droupout changed to 0.1       
#     nn.Linear(512, 512),
#     nn.ReLU(inplace=True),
#     nn.Dropout(0.1),  
#     nn.Linear(512, 512),
#     nn.ReLU(inplace=True),
#     nn.Dropout(0.1),  
#     nn.Linear(512, 512),
#     nn.ReLU(inplace=True),
#     nn.Dropout(0.1),        
#     nn.Linear(512, num_classes)
# ).to(device)


##  then we choose 512 vs 1024 vs if any - then experiment with number of layers (1, 3, 5)

# Freeze CLIP visual encoder - only train the classifier
for param in model.visual.parameters():
    param.requires_grad = True


def train_one_epoch(criterion, optimizer ):
    model.train()  # Set model to eval mode to disable dropout and batch norm updates
    classifier.train()

    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in tqdm(enumerate(train_loader)):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() 
        # with torch.no_grad():
        image_features = model.encode_image(images) # we only use the clip image encoder and ignore the text encoder since we have only images and no text pairs

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # added normalization

        logits = classifier(image_features)
        loss = criterion(logits, labels) # loss per batch 

        # finding batch_size 
        bs = labels.size(0) # to avoid the last batch size being smaller inconsistency
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.visual.parameters()) + list(classifier.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        total_loss += loss.item() * bs # prev: for one batch + one batch / total batch len(trai_loader) --- new: for each sample in the batch (512)+ each sample in the batch  (512)+ 200  == all samples loss / (all samples)
        _, preds = logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += bs # number of all samples (eg 512+512+... + 200)

        # LOG EVERY 10 STEPS - you can change this 10 to any number you like
        if (batch_idx + 1) % 10 == 0:
            wandb.log({
                "train/train_loss": loss.item(),
                "train/train_acc_step": correct / total,
                "train/running_loss": total_loss / total,
                "train/step": batch_idx + 1
            })
            print(
                f"[EPOCH {epoch}]:"
                f"Train Step [{batch_idx+1}/{len(train_loader)}] "
                f"Batch Loss: {loss.item():.4f} "
                f"Train Acc: {correct/total:.4f}"
            )
    return total_loss / total, correct / total




def evaluate(loader):
    model.eval()  # CLIP model stays in eval mode
    classifier.eval()

    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device) # sending images and labels to the gpu
            image_features = model.encode_image(images) # we call only the image encoder for our task 
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) # normalization - tocheck
            logits = classifier(image_features)

            _, preds = logits.max(1) 
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            # Accumulate predictions and labels for metric computation
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

        # Compute metrics over entire dataset
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        acc = correct / total
        f1 = f1_score(all_labels, all_preds, zero_division=0, average='weighted')
        precision = precision_score(all_labels, all_preds, zero_division=0, average='weighted')
        recall = recall_score(all_labels, all_preds, zero_division=0, average='weighted')

    return acc, f1, precision, recall


print("Training Begins...")
EPOCHS = 10



# making class weights for weighted cross entropy
labels = [label for _, label in train_dataset.samples]
class_counts = Counter(labels)
total = len(labels)
class_weights = [total / (num_classes * class_counts[i]) for i in range(num_classes)] # taking inverse of it 
class_weights = torch.tensor(class_weights, dtype = torch.float).to(device)
print("Class counts:", class_counts)
print("Class weights:", class_weights)


optimizer = torch.optim.AdamW(
    [
        {"params": classifier.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
        {"params": model.visual.parameters(), "lr": 2e-6, "weight_decay": 1e-4},
    ]
)

# loss definition
label_smoothing = 0.1
if loss_type=="cross_entropy_woLS":
    criterion = nn.CrossEntropyLoss()
    print("[INFO] Using Cross entropy loss") 
elif loss_type=="cross_entropy_withLS":
    criterion = nn.CrossEntropyLoss(label_smoothing = label_smoothing)
    print("[INFO] Using CE with label smoothening") 
elif loss_type=="weighted_crossEntropy":
    criterion  = nn.CrossEntropyLoss(weight=class_weights)
else:
    print("[ERROR] no loss found")

# scheduler definition
if scheduler == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
elif scheduler == "StepLR":
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5,gamma=0.1)
else: 
    print("[ERROR] no scheduler found")


for epoch in range(EPOCHS):
    train_loss, train_acc = train_one_epoch(criterion=criterion, optimizer=optimizer) # train acc = real train acc

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Train Acc:  {train_acc:.4f}")
    
    # real test acc
    real_test_acc, real_test_f1, real_test_precision, real_test_recall = evaluate(real_test_loader)
    print(f"  Source Test Acc (Real): {real_test_acc:.4f}")
    print(f"  Source Test F1 (Real): {real_test_f1:.4f}")
    print(f"  Source Test Precision (Real): {real_test_precision:.4f}")
    print(f"  Source Test Recall (Real): {real_test_recall:.4f}")

    # infograph test acc 
    infograph_test_acc, infograph_test_f1, infograph_test_precision, infograph_test_recall= evaluate(infograph_test_loader)
    print(f"  Target Test Acc (Infograph): {infograph_test_acc:.4f}")
    print(f"  Target Test F1 (Infograph): {infograph_test_f1:.4f}")
    print(f"  Target Test Precision (Infograph): {infograph_test_precision:.4f}")
    print(f"  Target Test Recall (Infograph): {infograph_test_recall:.4f}")

    # clipart test acc 
    clipart_test_acc, clipart_test_f1, clipart_test_precision, clipart_test_recall= evaluate(clipart_test_loader)
    print(f"  Target Test Acc (Clipart): {clipart_test_acc:.4f}")
    print(f"  Target Test F1 (Clipart): {clipart_test_f1:.4f}")
    print(f"  Target Test Precision (Clipart): {clipart_test_precision:.4f}")
    print(f"  Target Test Recall (Clipart): {clipart_test_recall:.4f}")


    wandb.log({
        "epoch": epoch + 1,
        "train/loss_epoch": train_loss,
        "train/acc_epoch": train_acc,
        "test/real_acc": real_test_acc,
        "test/infograph_acc": infograph_test_acc,
        "test/clipart_acc": clipart_test_acc,

        "test/real_f1": real_test_f1,
        "test/infograph_f1": infograph_test_f1,
        "test/clipart_f1": clipart_test_f1,

        "test/real_precision": real_test_precision,
        "test/infograph_precision": infograph_test_precision,
        "test/clipart_precision": clipart_test_precision,

        "test/real_recall": real_test_recall,
        "test/infograph_recall": infograph_test_recall,
        "test/clipart_recall": clipart_test_recall,

    })
    scheduler.step()
    

# save_path = "models/10epochs_3fclayer_0.01lr_run.pth"
# torch.save(
#     {
#         "clip_visual_state_dict": model.visual.state_dict(),
#         "classifier_state_dict": classifier.state_dict(),
#         "num_classes": num_classes,
#     }, model_name)

# print(f"\nModel saved to: {model_name}")
wandb.finish()


# todo: save model with .p file 
# todo: add f1 score (precision and recall)  - log in wandb - print
# todo: check why loss is so high


# todo: check the training
# todo: check if we should do normalization or no 
# todo: visualization - tsne 



##### (a) ensemble methods - with and without diveristy loss
#### (b) normal finetuning - (Hidden layer 256, 512, 1024), (MLP Layers - 1, 2, 5), (Parametes- LR (0.01, 0.001, 0.0001), Scheduler (LR, Cosine), Loss - (CE, wCE), A epoch graph showing no increase in performance and thereby justifying the choice behind choosing 10 epochs)
#### ## tsne - before after
#### domain generlllization gap 

