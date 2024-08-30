import os
from typing import List

import cv2
import numpy as np
import numpy.typing as npt
from moviepy.editor import VideoFileClip, clips_array
from tqdm import tqdm

NDArrayByte = npt.NDArray[np.uint8]

def layout_nuplan(
    imgs: List[np.ndarray], cam_names: List[str]
) -> np.ndarray:
    """Combine cameras into a tiled image for NuPlan dataset."""
    channel = imgs[0].shape[-1]
    height, width = imgs[0].shape[:2]

    tiled_height = height * 3
    tiled_width = width * 3
    tiled_img = np.zeros((tiled_height, tiled_width, channel), dtype=np.uint8)
    filled_mask = np.zeros((tiled_height, tiled_width), dtype=np.uint8)
    
    for idx, cam_name in enumerate(cam_names):
        img = imgs[idx]
        if cam_name == "CAM_F0":
            tiled_img[:height, width:2*width] = img
            filled_mask[:height, width:2*width] = 1
        elif cam_name == "CAM_L0":
            tiled_img[:height, :width] = img
            filled_mask[:height, :width] = 1
        elif cam_name == "CAM_R0":
            tiled_img[:height, 2*width:] = img
            filled_mask[:height, 2*width:] = 1
        elif cam_name == "CAM_L1":
            tiled_img[height:2*height, :width] = img
            filled_mask[height:2*height, :width] = 1
        elif cam_name == "CAM_R1":
            tiled_img[height:2*height, 2*width:] = img
            filled_mask[height:2*height, 2*width:] = 1
        elif cam_name == "CAM_L2":
            tiled_img[2*height:, :width] = img
            filled_mask[2*height:, :width] = 1
        elif cam_name == "CAM_R2":
            tiled_img[2*height:, 2*width:] = img
            filled_mask[2*height:, 2*width:] = 1
        elif cam_name == "CAM_B0":
            tiled_img[2*height:, width:2*width] = img
            filled_mask[2*height:, width:2*width] = 1

    # Crop the image according to the largest filled area
    min_y, max_y = np.where(filled_mask)[0].min(), np.where(filled_mask)[0].max()
    min_x, max_x = np.where(filled_mask)[1].min(), np.where(filled_mask)[1].max()
    tiled_img = tiled_img[min_y:max_y+1, min_x:max_x+1]
    
    return tiled_img

def preview_instances(
    input_root: str = 'data/nuplan/processed/mini',
    output_root: str = 'data/nuplan/preview/instances',
    frame_width: int = 1920,
    frame_height: int = 1080,
    fps: int = 24,
):
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Iterate over each scene directory in the input root directory
    for scene_id in tqdm(sorted(os.listdir(input_root))):
        scene_folder = os.path.join(input_root, scene_id, "instances", "debug_vis")
        
        # Define the output video file path
        output_video_path = os.path.join(output_root, f'{scene_id}.mp4')
        
        # Get paths to each camera image folder
        images_all = sorted([f for f in sorted(os.listdir(scene_folder)) if f.lower().endswith('.jpg')])
        num_images = len(images_all)
        num_frames = num_images // 8  # NuPlan has 8 cameras
        
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        for frame_id in range(num_frames):
            imgs = []
            cam_names = []
            for cam_id in range(8):
                img = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 8 + cam_id]))
                imgs.append(img)
                cam_names.append(f"CAM_{['F0', 'L0', 'R0', 'L1', 'R1', 'L2', 'R2', 'B0'][cam_id]}")
            
            combined_img = layout_nuplan(imgs, cam_names)
            combined_img = cv2.resize(combined_img, (frame_width, frame_height))
            
            out.write(combined_img)
        out.release()

def preview_camera_captures(
    input_root: str = 'data/nuplan/processed/mini',
    output_root: str = 'data/nuplan/preview/camera_captures',
    frame_width: int = 1920,
    frame_height: int = 1080,
    fps: int = 24
):
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Iterate over each scene directory in the input root directory
    for scene_id in tqdm(sorted(os.listdir(input_root))):
        scene_folder = os.path.join(input_root, scene_id, "images")
        
        # Define the output video file path
        output_video_path = os.path.join(output_root, f'{scene_id}.mp4')
        
        # Get paths to each camera image folder
        images_all = sorted([f for f in sorted(os.listdir(scene_folder)) if f.lower().endswith('.jpg')])
        num_images = len(images_all)
        num_frames = num_images // 8  # NuPlan has 8 cameras
        
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        for frame_id in range(num_frames):
            imgs = []
            cam_names = []
            for cam_id in range(8):
                img = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 8 + cam_id]))
                imgs.append(img)
                cam_names.append(f"CAM_{['F0', 'L0', 'R0', 'L1', 'R1', 'L2', 'R2', 'B0'][cam_id]}")
            
            combined_img = layout_nuplan(imgs, cam_names)
            combined_img = cv2.resize(combined_img, (frame_width, frame_height))
            
            out.write(combined_img)
        out.release()

def preview_dynamic_masks(
    input_root: str = 'data/nuplan/processed/mini',
    output_root: str = 'data/nuplan/preview/dynamic_masks',
    frame_width: int = 1920,
    frame_height: int = 1080,
    fps: int = 24,
    dynamic_class: str = 'all' # 'all' or 'vehicle' or 'human'
):
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Iterate over each scene directory in the input root directory
    for scene_id in tqdm(sorted(os.listdir(input_root))):
        scene_folder = os.path.join(input_root, scene_id, "dynamic_masks", dynamic_class)
        
        # Define the output video file path
        output_video_path = os.path.join(output_root, f'{scene_id}.mp4')
        
        # Get paths to each camera image folder
        images_all = sorted([f for f in sorted(os.listdir(scene_folder)) if f.lower().endswith('.png')])
        num_images = len(images_all)
        num_frames = num_images // 8  # NuPlan has 8 cameras
        
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        for frame_id in range(num_frames):
            imgs = []
            cam_names = []
            for cam_id in range(8):
                img = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 8 + cam_id]))
                imgs.append(img)
                cam_names.append(f"CAM_{['F0', 'L0', 'R0', 'L1', 'R1', 'L2', 'R2', 'B0'][cam_id]}")
            
            combined_img = layout_nuplan(imgs, cam_names)
            combined_img = cv2.resize(combined_img, (frame_width, frame_height))
            
            out.write(combined_img)
        out.release()

def cat_all_videos(
    keys: list,
    output_root: str = 'data/nuplan/preview/catted_videos',
    scene_id: str = 'all',
    fps: int = 24
):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    if scene_id == 'all':
        scene_ids = sorted(os.listdir('data/nuplan/processed/mini'))
    else:
        scene_ids = [scene_id]
        
    for scene_id in scene_ids:
        video_paths = []
        for key in keys:
            video_paths.append(f'data/nuplan/preview/{key}/{scene_id}.mp4')
        
        output_video_path = os.path.join(output_root, f'{scene_id}.mp4')

        # use moviepy to concatenate videos
        clips = [VideoFileClip(path) for path in video_paths]
        # cat videos from top to bottom, vertically
        final_clip = clips_array([[clip] for clip in clips])
        final_clip.write_videofile(output_video_path, fps=fps)
        final_clip.close()
            
if __name__ == '__main__':
    # preview_camera_captures(fps=24)
    # preview_dynamic_masks(fps=24)
    preview_instances(fps=24)
    # cat_all_videos(keys=['instances'], scene_id='all', fps=24)