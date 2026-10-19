"""Model definitions.

The classifier is split into a backbone that produces an embedding, an optional
BNNeck, and a head.  The default head is ArcFace (additive angular margin); CosFace
and a plain linear softmax head are also available so the three can be compared.

With a margin head the model's forward pass needs the labels in order to apply the
margin to the ground-truth logit only::

    logits = model(images, labels)   # training
    logits = model(images)           # inference (labels=None -> scale * cosine)

Both cases return logits, so the training loop can keep using CrossEntropyLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34


class MarginHead(nn.Module):
    """ArcFace / CosFace classifier over L2-normalised embeddings and class centres."""

    SUPPORTED = ("arcface", "cosface")

    def __init__(
        self,
        in_features,
        num_classes,
        loss_type="arcface",
        margin=0.35,
        scale=30.0,
        margin_warmup_epochs=0,
    ):
        super().__init__()
        if loss_type not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported margin loss '{loss_type}'. "
                f"Choose one of {self.SUPPORTED}."
            )
        if margin <= 0:
            raise ValueError("margin must be positive.")
        if scale <= 0:
            raise ValueError("scale must be positive.")

        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.loss_type = loss_type
        self.base_margin = float(margin)
        self.margin = float(margin)
        self.scale = float(scale)
        self.margin_warmup_epochs = int(margin_warmup_epochs)

    def set_epoch(self, epoch: int) -> None:
        """Ramp the margin in over the first epochs to stabilise early training."""
        if self.margin_warmup_epochs > 0:
            progress = min(1.0, (epoch + 1) / self.margin_warmup_epochs)
            self.margin = self.base_margin * progress

    def forward(self, embeddings, labels=None):
        cosine = F.linear(
            F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1)
        )
        if labels is None:
            return self.scale * cosine

        if self.loss_type == "arcface":
            # cos(theta + m) expands to cos(theta)cos(m) - sin(theta)sin(m);
            # acos/sin keeps the gradient well defined at the clamp boundaries.
            cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            theta = torch.acos(cosine)
            target_logit = torch.cos(theta + self.margin)
        else:  # cosface
            target_logit = cosine - self.margin

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = cosine * (1.0 - one_hot) + target_logit * one_hot
        return self.scale * logits


class FaceClassifier(nn.Module):
    """Backbone -> BNNeck -> (ArcFace | CosFace | linear) head."""

    def __init__(
        self,
        backbone,
        embed_dim,
        num_classes,
        head="arcface",
        margin=0.35,
        scale=30.0,
        bnneck=True,
        margin_warmup_epochs=0,
    ):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.head_type = head
        self.neck = nn.BatchNorm1d(embed_dim) if bnneck else nn.Identity()

        if head == "linear":
            self.head = nn.Linear(embed_dim, num_classes)
        elif head in MarginHead.SUPPORTED:
            self.head = MarginHead(
                embed_dim,
                num_classes,
                loss_type=head,
                margin=margin,
                scale=scale,
                margin_warmup_epochs=margin_warmup_epochs,
            )
        else:
            raise ValueError(
                f"Unsupported head '{head}'. Choose 'arcface', 'cosface' or 'linear'."
            )

    def forward(self, x, labels=None):
        features = self.neck(self.backbone(x))
        if self.head_type == "linear":
            return self.head(features)
        return self.head(features, labels)

    @torch.no_grad()
    def embed(self, x):
        """L2-normalised embedding, e.g. for prototype/kNN inference."""
        return F.normalize(self.neck(self.backbone(x)), dim=1)

    def set_epoch(self, epoch: int) -> None:
        if isinstance(self.head, MarginHead):
            self.head.set_epoch(epoch)


def _resnet_backbone(builder, dropout):
    """ResNet stem adapted to 64x64 inputs.

    A 3x3/stride-1 first convolution plus no max-pool keeps enough spatial detail
    for faces at 64 px: the last stage still sees an 8x8 feature map instead of 2x2.
    """
    model = builder(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Identity()
    model.avgpool = nn.Sequential(model.avgpool, nn.Dropout(dropout))
    return model, 512


_BACKBONES = {
    "resnet18": resnet18,
    "resnet34": resnet34,
}


def _simple_cnn_backbone(dropout):
    return CNN(dropout=dropout), 128


def get_model(
    model_name,
    num_classes=10,
    dropout=0.5,
    head="arcface",
    margin=0.35,
    scale=30.0,
    bnneck=True,
    margin_warmup_epochs=0,
):
    if model_name == "simpleCNN":
        backbone, embed_dim = _simple_cnn_backbone(dropout)
    elif model_name in _BACKBONES:
        backbone, embed_dim = _resnet_backbone(_BACKBONES[model_name], dropout)
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Choose one of: simpleCNN, {', '.join(sorted(_BACKBONES))}."
        )

    return FaceClassifier(
        backbone,
        embed_dim,
        num_classes,
        head=head,
        margin=margin,
        scale=scale,
        bnneck=bnneck,
        margin_warmup_epochs=margin_warmup_epochs,
    )


class CNN(torch.nn.Module):
    """Small convolutional embedding network (used with model_name='simpleCNN')."""

    def __init__(self, dropout=0.5, embedding_dim=128):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = torch.nn.Linear(64 * 4 * 4, embedding_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.dropout(self.relu(self.fc1(x)))
