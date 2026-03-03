import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from PIL import Image


class GradientReversal(torch.autograd.Function):
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


def load_backend_model(
    checkpoint_path: str,
    class_names: list[str],
    clip_model_name: str = "ViT-B-32",
    clip_pretrained: str = "openai",
    device: str = "cpu",
):
    device_t = torch.device(device)
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )
    clip_model = clip_model.to(device_t).eval()

    feature_dim = clip_model.visual.output_dim
    dann = DANNHead(feature_dim=feature_dim, num_classes=len(class_names)).to(device_t).eval()

    ckpt = torch.load(checkpoint_path, map_location=device_t)
    clip_model.visual.load_state_dict(ckpt["clip_visual_state_dict"], strict=True)
    dann.load_state_dict(ckpt["dann_state_dict"], strict=True)

    return clip_model, dann, preprocess, device_t


@torch.inference_mode()
def predict_topk(clip_model, dann, preprocess, device, pil_image: Image.Image, class_names, topk=1):
    x = preprocess(pil_image).unsqueeze(0).to(device)
    feats = clip_model.encode_image(x).float()
    feats = F.normalize(feats, dim=-1)

    logits, _ = dann(feats, lambda_=0.0)
    probs = torch.softmax(logits[0], dim=-1)

    k = min(topk, probs.numel())
    top_probs, top_idx = torch.topk(probs, k=k)

    out = []
    for p, idx in zip(top_probs.tolist(), top_idx.tolist()):
        out.append({"label": class_names[int(idx)], "prob": float(p), "class_id": int(idx)})
    return out