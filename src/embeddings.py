from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import open_clip

from .utils import pil_rgb


def build_dino_transform(img_size: int = 224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_dino_model(device: torch.device, model_name: str = "dinov2_vitb14"):
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    return model


def load_clip_model(device: torch.device, model_name: str = "ViT-H-14", pretrained: str = "laion2b_s32b_b79k"):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    return model, preprocess


def batchify(items: list, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


@torch.no_grad()
def extract_embeddings_unified(
    paths: list[str],
    dino_model,
    clip_model,
    clip_preprocess,
    dino_transform,
    device: torch.device,
    batch_size: int = 16,
    tta: bool = True,
):
    dino_all, clip_all = [], []
    for batch_paths in tqdm(list(batchify(paths, batch_size)), desc="Extracting DINO & CLIP"):
        d_imgs, c_imgs, d_imgs_f, c_imgs_f = [], [], [], []
        for p in batch_paths:
            img = pil_rgb(p)
            d_imgs.append(dino_transform(img))
            c_imgs.append(clip_preprocess(img))
            if tta:
                img_f = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                d_imgs_f.append(dino_transform(img_f))
                c_imgs_f.append(clip_preprocess(img_f))

        dx = torch.stack(d_imgs).to(device)
        d_emb = F.normalize(dino_model(dx), dim=-1)
        if tta:
            dxf = torch.stack(d_imgs_f).to(device)
            d_emb_f = F.normalize(dino_model(dxf), dim=-1)
            d_emb = F.normalize((d_emb + d_emb_f) / 2.0, dim=-1)
        dino_all.append(d_emb.cpu().numpy())

        cx = torch.stack(c_imgs).to(device)
        c_emb = F.normalize(clip_model.encode_image(cx), dim=-1)
        if tta:
            cxf = torch.stack(c_imgs_f).to(device)
            c_emb_f = F.normalize(clip_model.encode_image(cxf), dim=-1)
            c_emb = F.normalize((c_emb + c_emb_f) / 2.0, dim=-1)
        clip_all.append(c_emb.cpu().numpy())

    return np.concatenate(dino_all, axis=0), np.concatenate(clip_all, axis=0)


@torch.no_grad()
def extract_single_embeddings(path, dino_model, clip_model, clip_preprocess, dino_transform, device, tta: bool = True):
    img = pil_rgb(path)
    dx = dino_transform(img).unsqueeze(0).to(device)
    d_emb = F.normalize(dino_model(dx), dim=-1)

    cx = clip_preprocess(img).unsqueeze(0).to(device)
    c_emb = F.normalize(clip_model.encode_image(cx), dim=-1)

    if tta:
        img_f = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        dxf = dino_transform(img_f).unsqueeze(0).to(device)
        d_emb_f = F.normalize(dino_model(dxf), dim=-1)
        d_emb = F.normalize((d_emb + d_emb_f) / 2.0, dim=-1)

        cxf = clip_preprocess(img_f).unsqueeze(0).to(device)
        c_emb_f = F.normalize(clip_model.encode_image(cxf), dim=-1)
        c_emb = F.normalize((c_emb + c_emb_f) / 2.0, dim=-1)

    return d_emb.cpu().numpy(), c_emb.cpu().numpy()
