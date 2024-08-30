"""
@file   extract_masks.py
@author Jianfei Guo, Shanghai AI Lab
@brief  Extract semantic mask

Using SegFormer, 2021. Cityscapes 83.2%
Relies on timm==0.3.2 & pytorch 1.8.1 (buggy on pytorch >= 1.9)

Installation:
    NOTE: mmcv-full==1.2.7 requires another pytorch version & conda env.
        Currently mmcv-full==1.2.7 does not support pytorch>=1.9; 
            will raise AttributeError: 'super' object has no attribute '_specify_ddp_gpu_num'
        Hence, a seperate conda env is needed.

    git clone https://github.com/NVlabs/SegFormer

    conda create -n segformer python=3.8
    conda activate segformer
    # conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

    pip install timm==0.3.2 pylint debugpy opencv-python attrs ipython tqdm imageio scikit-image omegaconf
    pip install mmcv-full==1.2.7 --no-cache-dir
    
    cd SegFormer
    pip install .

Usage:
    Direct run this script in the newly set conda env.
"""


from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette

semantic_classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]
dataset_classes_in_sematic = {
    'Vehicle': [13, 14, 15],   # 'car', 'truck', 'bus'
    'human': [11, 12, 17, 18], # 'person', 'rider', 'motorcycle', 'bicycle'
}

if __name__ == "__main__":
    import os
    import imageio
    import numpy as np
    from glob import glob
    from tqdm import tqdm
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # Custom configs
    parser.add_argument('--data_root', type=str, default='data/waymo/processed/training')
    parser.add_argument(
        "--scene_ids",
        default=None,
        type=int,
        nargs="+",
        help="scene ids to be processed, a list of integers separated by space. Range: [0, 798] for training, [0, 202] for validation",
    )
    parser.add_argument(
        "--split_file", type=str, default=None, help="Split file in data/waymo_splits"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="If no scene id or split_file is given, use start_idx and num_scenes to generate scene_ids_list",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=200,
        help="number of scenes to be processed",
    )
    parser.add_argument(
        '--process_dynamic_mask',
        action='store_true',
        help="Whether to process dynamic masks",
    )
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ignore_existing', action='store_true')
    parser.add_argument('--no_compress', action='store_true')
    parser.add_argument('--rgb_dirname', type=str, default="images")
    parser.add_argument('--mask_dirname', type=str, default="fine_dynamic_masks")

    # Algorithm configs
    parser.add_argument('--segformer_path', type=str, default='/home/guojianfei/ai_ws/SegFormer')
    parser.add_argument('--config', help='Config file', type=str, default=None)
    parser.add_argument('--checkpoint', help='Checkpoint file', type=str, default=None)
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--palette', default='cityscapes', help='Color palette used for segmentation map')
    
    args = parser.parse_args()
    if args.config is None:
        args.config = os.path.join(args.segformer_path, 'local_configs', 'segformer', 'B5', 'segformer.b5.1024x1024.city.160k.py')
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.segformer_path, 'pretrained', 'segformer.b5.1024x1024.city.160k.pth')
    
    if args.scene_ids is not None:
        scene_ids_list = args.scene_ids
    elif args.split_file is not None:
        # parse the split file
        split_file = open(args.split_file, "r").readlines()[1:]
        # NOTE: small hack here, to be refined in the futher (TODO)
        if "kitti" in args.split_file or "nuplan" in args.split_file:
            scene_ids_list = [line.strip().split(",")[0] for line in split_file]
        else:
            scene_ids_list = [int(line.strip().split(",")[0]) for line in split_file]
    else:
        scene_ids_list = np.arange(args.start_idx, args.start_idx + args.num_scenes)
    
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    
    for scene_i, scene_id in enumerate(tqdm(scene_ids_list, f'Extracting Masks ...')):
        scene_id = str(scene_id).zfill(3)
        img_dir = os.path.join(args.data_root, scene_id, args.rgb_dirname)
        
        # create mask dir
        sky_mask_dir = os.path.join(args.data_root, scene_id, "sky_masks")
        if not os.path.exists(sky_mask_dir):
            os.makedirs(sky_mask_dir)
        
        # create dynamic mask dir
        if args.process_dynamic_mask:
            rough_human_mask_dir = os.path.join(args.data_root, scene_id, "dynamic_masks", "human")
            rough_vehicle_mask_dir = os.path.join(args.data_root, scene_id, "dynamic_masks", "vehicle")
            
            all_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "all")
            if not os.path.exists(all_mask_dir):
                os.makedirs(all_mask_dir)
            human_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "human")
            if not os.path.exists(human_mask_dir):
                os.makedirs(human_mask_dir)
            vehicle_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "vehicle")
            if not os.path.exists(vehicle_mask_dir):
                os.makedirs(vehicle_mask_dir)
        
        flist = sorted(glob(os.path.join(img_dir, '*')))
        for fpath in tqdm(flist, f'scene[{scene_id}]'):
            fbase = os.path.splitext(os.path.basename(os.path.normpath(fpath)))[0]
    
            # if args.no_compress:
            #     mask_fpath = os.path.join(mask_dir, f"{fbase}.npy")
            # else:
            #     mask_fpath = os.path.join(mask_dir, f"{fbase}.npz")
            
            if args.ignore_existing and os.path.exists(os.path.join(args.data_root, scene_id, "fine_dynamic_masks")):
                continue
            
            #---- Inference and save outputs
            result = inference_segmentor(model, fpath)
            mask = result[0].astype(np.uint8)   # NOTE: in the settings of "cityscapes", there are 19 classes at most.
            # if args.no_compress:
            #     np.save(mask_fpath, mask)
            # else:
            #     np.savez_compressed(mask_fpath, mask)   # NOTE: compressed files are 100x smaller.

            # save sky mask
            sky_mask = np.isin(mask, [10])
            imageio.imwrite(os.path.join(sky_mask_dir, f"{fbase}.png"), sky_mask.astype(np.uint8)*255)
            
            if args.process_dynamic_mask:
                # save human masks
                rough_human_mask_path = os.path.join(rough_human_mask_dir, f"{fbase}.png")
                rough_human_mask = (imageio.imread(rough_human_mask_path) > 0)
                huamn_mask = np.isin(mask, dataset_classes_in_sematic['human'])
                valid_human_mask = np.logical_and(huamn_mask, rough_human_mask)
                imageio.imwrite(os.path.join(human_mask_dir, f"{fbase}.png"), valid_human_mask.astype(np.uint8)*255)
                
                # save vehicle mask
                rough_vehicle_mask_path = os.path.join(rough_vehicle_mask_dir, f"{fbase}.png")
                rough_vehicle_mask = (imageio.imread(rough_vehicle_mask_path) > 0)
                vehicle_mask = np.isin(mask, dataset_classes_in_sematic['Vehicle'])
                valid_vehicle_mask = np.logical_and(vehicle_mask, rough_vehicle_mask)
                imageio.imwrite(os.path.join(vehicle_mask_dir, f"{fbase}.png"), valid_vehicle_mask.astype(np.uint8)*255)
                
                # save dynamic mask
                valid_all_mask = np.logical_or(valid_human_mask, valid_vehicle_mask)
                imageio.imwrite(os.path.join(all_mask_dir, f"{fbase}.png"), valid_all_mask.astype(np.uint8)*255)