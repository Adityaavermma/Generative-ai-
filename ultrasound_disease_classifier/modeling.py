"""Model + transforms shared by train.py and predict.py."""

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2 as T

from common import CLASSES

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class USNet(nn.Module):
    """Pretrained CNN backbone with two heads.

    * classification head: benign / malignant / normal
    * lesion head: a coarse lesion map predicted from the last feature map.
      During training it is supervised with the BUSI lesion masks (an
      auxiliary task that teaches the backbone to look at the lesion instead
      of image artefacts). At prediction time no mask is needed; the head's
      output is shown as a "where is the lesion" heat-map.
    """

    def __init__(self, arch: str, pretrained: bool = True, drop_path: float = 0.1):
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=len(CLASSES),
                                          drop_path_rate=drop_path)
        c = self.backbone.num_features
        self.seg_head = nn.Sequential(nn.Conv2d(c, 128, 3, padding=1), nn.GELU(), nn.Conv2d(128, 1, 1))

    def forward(self, x, with_seg: bool = False):
        f = self.backbone.forward_features(x)
        if f.ndim != 4:
            raise ValueError("USNet needs a CNN backbone that returns a BxCxHxW feature map")
        logits = self.backbone.forward_head(f)
        return (logits, self.seg_head(f)) if with_seg else logits


def build_model(arch: str, pretrained: bool = True, drop_path: float = 0.1):
    return USNet(arch, pretrained=pretrained, drop_path=drop_path)


def gpu_augment(x, m, size, scale=(0.6, 1.0), ratio=(0.8, 1.25), degrees=10.0,
                translate=0.05, brightness=0.2, contrast=0.2):
    """Batched training augmentation on the GPU.

    x: B x 1 x H x W float in [0, 1]; m: B x 1 x H x W lesion mask (0/1).
    Per sample: random resized crop + horizontal flip + rotation + shift,
    done as ONE affine warp applied identically to image and mask, then
    brightness / contrast jitter on the image only. No vertical flip: in
    ultrasound the transducer is always at the top, so depth direction
    carries meaning (e.g. posterior shadowing).

    (Doing this on the GPU instead of in DataLoader workers avoids ~2.5 min
    of worker start-up per fold on Windows.)
    Returns (normalised B x 3 x size x size image, B x 1 x size x size mask).
    """
    b, dev = x.shape[0], x.device

    def uniform(lo, hi):
        return torch.empty(b, device=dev).uniform_(lo, hi)

    # random resized crop: crop width/height as a fraction of the image
    area = uniform(*scale)
    r = torch.exp(uniform(math.log(ratio[0]), math.log(ratio[1])))
    cw, ch = torch.sqrt(area * r).clamp(max=1.0), torch.sqrt(area / r).clamp(max=1.0)
    cx = (torch.rand(b, device=dev) * (1 - cw) + cw / 2) * 2 - 1   # crop centre, [-1, 1]
    cy = (torch.rand(b, device=dev) * (1 - ch) + ch / 2) * 2 - 1

    ang = uniform(-degrees, degrees) * math.pi / 180
    flip = torch.where(torch.rand(b, device=dev) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(ang), torch.sin(ang)
    rot_flip = torch.stack([torch.stack([cos * flip, -sin], 1),
                            torch.stack([sin * flip, cos], 1)], 1)          # B x 2 x 2
    a = torch.diag_embed(torch.stack([cw, ch], 1)) @ rot_flip
    shift = torch.stack([uniform(-translate, translate), uniform(-translate, translate)], 1) * 2
    theta = torch.cat([a, (torch.stack([cx, cy], 1) - (a @ shift[..., None])[..., 0])[..., None]], 2)

    grid = F.affine_grid(theta, (b, 1, size, size), align_corners=False)
    xo = F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    mo = F.grid_sample(m.float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)

    xo = (xo * uniform(1 - brightness, 1 + brightness).view(b, 1, 1, 1)).clamp(0, 1)
    mean = xo.mean((2, 3), keepdim=True)
    xo = ((xo - mean) * uniform(1 - contrast, 1 + contrast).view(b, 1, 1, 1) + mean).clamp(0, 1)

    mean_t = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std_t = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return (xo.expand(-1, 3, -1, -1) - mean_t) / std_t, mo


def eval_transform(size: int):
    return T.Compose([
        T.Resize((size, size), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(MEAN, STD),
    ])


def gray_to_tensor(img):
    """uint8 HxW numpy -> uint8 1xHxW tensor. Call .expand(3, -1, -1) to get RGB."""
    return torch.from_numpy(img).unsqueeze(0)


def _autocast(x):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda)


@torch.no_grad()
def predict_proba(model, x, tta: bool = True):
    """Softmax probabilities, averaged with a horizontal-flip view."""
    with _autocast(x):
        probs = model(x).float().softmax(1)
        if tta:
            probs = (probs + model(torch.flip(x, dims=[3])).float().softmax(1)) / 2
    return probs


@torch.no_grad()
def predict_with_map(model, x):
    """Class probabilities and lesion map from the same two forward passes
    (original + horizontal flip) - half the cost of predict_proba + lesion_map."""
    with _autocast(x):
        logits, seg = model(x, with_seg=True)
        logits_f, seg_f = model(torch.flip(x, dims=[3]), with_seg=True)
    probs = (logits.float().softmax(1) + logits_f.float().softmax(1)) / 2
    m = (seg.float().sigmoid() + torch.flip(seg_f.float().sigmoid(), dims=[3])) / 2
    return probs, F.interpolate(m, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]


@torch.no_grad()
def lesion_map(model, x, tta: bool = True):
    """Lesion probability map, upsampled to the input size (B x H x W)."""
    with _autocast(x):
        m = model(x, with_seg=True)[1].float().sigmoid()
        if tta:
            mf = model(torch.flip(x, dims=[3]), with_seg=True)[1].float().sigmoid()
            m = (m + torch.flip(mf, dims=[3])) / 2
    return F.interpolate(m, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
