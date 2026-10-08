"""Masks for an UrbanV2X scene: sky, dynamic (semantic), and static ego-hood masks.

  sky_masks/{t:03d}_{cam}.png                 SegFormer (Cityscapes) 'sky'
  dynamic_masks/{all,vehicle,human}/...       Cityscapes vehicle / human classes. Semantic, so parked vehicles are
                                              included: used for the dynamic-region metrics, not for training.
  data/ego_masks/urbanv2x/{cam}.png           ego hood: region touching the bottom row whose pixels are static over the
                                              clip, saturated (bright hood under auto-exposure) or almost always 'car'
Usage:  python datasets/urbanv2x/urbanv2x_masks.py --scene_dir data/urbanv2x/processed/<scene> [--device cpu]
"""
import argparse, glob, os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

p = argparse.ArgumentParser()
p.add_argument("--scene_dir", required=True)
p.add_argument("--ego_mask_dir", default="data/ego_masks/urbanv2x")
p.add_argument("--model", default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
p.add_argument("--cams", type=int, nargs="*", default=None, help="default: all cameras present in images/")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()

proc = SegformerImageProcessor.from_pretrained(a.model)
model = SegformerForSemanticSegmentation.from_pretrained(a.model).to(a.device).eval()
lab = {v.lower(): int(k) for k, v in model.config.id2label.items()}
SKY = lab["sky"]; VEH = [lab[k] for k in ("car", "truck", "bus", "train", "motorcycle", "bicycle")]; HUM = [lab[k] for k in ("person", "rider")]
for d in ("sky_masks", "dynamic_masks/all", "dynamic_masks/vehicle", "dynamic_masks/human"):
    os.makedirs(f"{a.scene_dir}/{d}", exist_ok=True)
os.makedirs(a.ego_mask_dir, exist_ok=True)

files = sorted(glob.glob(f"{a.scene_dir}/images/*.jpg"))
if a.cams is not None:
    files = [f for f in files if int(os.path.basename(f)[:-4].split("_")[1]) in a.cams]
stack, carfreq = {}, {}
with torch.no_grad():
    for i, f in enumerate(files):
        name = os.path.basename(f)[:-4]; cam = int(name.split("_")[1])
        im = Image.open(f).convert("RGB"); W, H = im.size
        logits = model(**proc(images=im, return_tensors="pt").to(a.device)).logits
        seg = torch.nn.functional.interpolate(logits, (H, W), mode="bilinear").argmax(1)[0].cpu().numpy()
        save = lambda m, d: Image.fromarray((m * 255).astype(np.uint8)).save(f"{a.scene_dir}/{d}/{name}.png")
        v, h = np.isin(seg, VEH), np.isin(seg, HUM)
        save(seg == SKY, "sky_masks"); save(v | h, "dynamic_masks/all"); save(v, "dynamic_masks/vehicle"); save(h, "dynamic_masks/human")
        stack.setdefault(cam, []).append(cv2.resize(np.asarray(im), (W // 4, H // 4)).astype(np.float32))
        carfreq.setdefault(cam, []).append(cv2.resize(v.astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_NEAREST))
        if i % 200 == 0: print(f"{i}/{len(files)}", flush=True)

for cam, ims in stack.items():
    s = np.stack(ims); std = s.std(0).mean(-1); h, w = std.shape
    bright = np.median(s, 0).mean(-1) > 225
    still = ((std < 6) | (np.mean(carfreq[cam], 0) >= 0.8) | bright).astype(np.uint8)
    still[: int(h * 0.6)] = 0
    n, cc = cv2.connectedComponents(still); keep = np.zeros_like(still)
    for k in set(cc[-1][still[-1] > 0]) - {0}:
        if (cc == k).sum() > 0.002 * h * w: keep[cc == k] = 1
    keep = cv2.dilate(cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)), np.ones((5, 5), np.uint8))
    full = cv2.resize(keep, (W, H), interpolation=cv2.INTER_NEAREST)
    print("cam", cam, "ego-mask fraction", round(float(full.mean()), 4))
    if full.mean() > 0.002: Image.fromarray(full * 255).save(f"{a.ego_mask_dir}/{cam}.png")
