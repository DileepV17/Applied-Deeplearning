import torch
import open_clip
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained CLIP (ViT-B/32 is a common starting point)
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai'
)
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model = model.to(device)

# Suppose you already split data into source/target folders
# I will use infograhic and real as source and target for now
infograph_data = datasets.ImageFolder("data/infograph/infograph", transform=preprocess)
real_data = datasets.ImageFolder("data/real/real", transform=preprocess)

infograph_loader = DataLoader(infograph_data, batch_size=32, shuffle=True)
real_loader = DataLoader(real_data, batch_size=32, shuffle=False)

class_names = infograph_data.classes
text_inputs = tokenizer(class_names)


# evaluate on zero shot learning
def zero_shot_eval(dataloader):
    text_features = model.encode_text(tokenizer(class_names).to(device))
    text_features /= text_features.norm(dim=-1, keepdim=True)

    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            logits = (100.0 * image_features @ text_features.T)
            preds = logits.argmax(dim=-1)
            correct += (preds.cpu() == labels).sum().item()
            total += labels.size(0)

    return correct / total

print("Zero-shot on infograph_data:", zero_shot_eval(infograph_loader))
print("Zero-shot on real_data:", zero_shot_eval(real_loader))