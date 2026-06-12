"""
@file   extract_masks.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Extract sky and fine dynamic masks.

Backend:
    Hugging Face SegFormer Cityscapes checkpoint, which works with the main
    training environment and modern GPUs.
"""

import os
from argparse import ArgumentParser
from glob import glob

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

semantic_classes = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

dataset_classes_in_sematic = {
    "Vehicle": [13, 14, 15],
    "human": [11, 12, 17, 18],
}


def parse_scene_ids(args):
    if args.scene_ids is not None:
        return args.scene_ids
    if args.split_file is not None:
        split_file = open(args.split_file, "r").readlines()[1:]
        if "kitti" in args.split_file or "nuplan" in args.split_file:
            return [line.strip().split(",")[0] for line in split_file]
        return [int(line.strip().split(",")[0]) for line in split_file]
    return np.arange(args.start_idx, args.start_idx + args.num_scenes)


class HuggingFaceSegmentationBackend:
    def __init__(self, model_id: str, device: str):
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        logits = self.model(**inputs).logits
        logits = F.interpolate(
            logits,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )
        mask = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
        return mask


def build_backend(args):
    return HuggingFaceSegmentationBackend(args.hf_model_id, args.device)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def should_skip(args, scene_id: str, fbase: str) -> bool:
    if not args.ignore_existing:
        return False

    sky_mask_path = os.path.join(args.data_root, scene_id, "sky_masks", f"{fbase}.png")
    if args.process_dynamic_mask:
        all_mask_path = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "all", f"{fbase}.png")
        return os.path.exists(sky_mask_path) and os.path.exists(all_mask_path)
    return os.path.exists(sky_mask_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/waymo/processed/training")
    parser.add_argument("--scene_ids", default=None, type=int, nargs="+")
    parser.add_argument("--split_file", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_scenes", type=int, default=200)
    parser.add_argument("--process_dynamic_mask", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ignore_existing", action="store_true")
    parser.add_argument("--no_compress", action="store_true")
    parser.add_argument("--rgb_dirname", type=str, default="images")
    parser.add_argument("--mask_dirname", type=str, default="fine_dynamic_masks")

    parser.add_argument(
        "--hf_model_id",
        type=str,
        default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        help="Hugging Face segmentation model id. B0 is the default for speed.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")

    args = parser.parse_args()
    scene_ids_list = parse_scene_ids(args)
    backend = build_backend(args)

    for scene_id in tqdm(scene_ids_list, "Extracting Masks ..."):
        scene_id = str(scene_id).zfill(3)
        img_dir = os.path.join(args.data_root, scene_id, args.rgb_dirname)

        sky_mask_dir = os.path.join(args.data_root, scene_id, "sky_masks")
        ensure_dir(sky_mask_dir)

        if args.process_dynamic_mask:
            rough_human_mask_dir = os.path.join(args.data_root, scene_id, "dynamic_masks", "human")
            rough_vehicle_mask_dir = os.path.join(args.data_root, scene_id, "dynamic_masks", "vehicle")

            all_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "all")
            human_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "human")
            vehicle_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "vehicle")
            ensure_dir(all_mask_dir)
            ensure_dir(human_mask_dir)
            ensure_dir(vehicle_mask_dir)

        flist = sorted(glob(os.path.join(img_dir, "*")))
        for fpath in tqdm(flist, f"scene[{scene_id}]"):
            fbase = os.path.splitext(os.path.basename(os.path.normpath(fpath)))[0]

            if should_skip(args, scene_id, fbase):
                continue

            mask = backend.predict(fpath)

            sky_mask = np.isin(mask, [10])
            imageio.imwrite(os.path.join(sky_mask_dir, f"{fbase}.png"), sky_mask.astype(np.uint8) * 255)

            if args.process_dynamic_mask:
                rough_human_mask_path = os.path.join(rough_human_mask_dir, f"{fbase}.png")
                rough_human_mask = imageio.imread(rough_human_mask_path) > 0
                human_mask = np.isin(mask, dataset_classes_in_sematic["human"])
                valid_human_mask = np.logical_and(human_mask, rough_human_mask)
                imageio.imwrite(
                    os.path.join(human_mask_dir, f"{fbase}.png"),
                    valid_human_mask.astype(np.uint8) * 255,
                )

                rough_vehicle_mask_path = os.path.join(rough_vehicle_mask_dir, f"{fbase}.png")
                rough_vehicle_mask = imageio.imread(rough_vehicle_mask_path) > 0
                vehicle_mask = np.isin(mask, dataset_classes_in_sematic["Vehicle"])
                valid_vehicle_mask = np.logical_and(vehicle_mask, rough_vehicle_mask)
                imageio.imwrite(
                    os.path.join(vehicle_mask_dir, f"{fbase}.png"),
                    valid_vehicle_mask.astype(np.uint8) * 255,
                )

                valid_all_mask = np.logical_or(valid_human_mask, valid_vehicle_mask)
                imageio.imwrite(
                    os.path.join(all_mask_dir, f"{fbase}.png"),
                    valid_all_mask.astype(np.uint8) * 255,
                )
