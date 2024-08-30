import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from moviepy.editor import VideoFileClip, clips_array
from typing import Mapping, Optional, Union
import numpy.typing as npt

NDArrayByte = npt.NDArray[np.uint8]

def tile_cameras(
    named_sensors: Mapping[str, Union[NDArrayByte, pd.DataFrame]],
    bev_img: Optional[NDArrayByte] = None,
) -> NDArrayByte:
    """Combine ring cameras into a tiled image.

    NOTE: Images are expected in BGR ordering.

    Layout:

        ##########################################################
        # ring_front_left # ring_front_center # ring_front_right #
        ##########################################################
        # ring_side_left  #                   #  ring_side_right #
        ##########################################################
        ############ ring_rear_left # ring_rear_right ############
        ##########################################################

    Args:
        named_sensors: Dictionary of camera names to the (width, height, 3) images.
        bev_img: (H,W,3) Bird's-eye view image.

    Returns:
        Tiled image.
    """
    landscape_height = 2048
    landscape_width = 1550
    for _, v in named_sensors.items():
        landscape_width = max(v.shape[0], v.shape[1])
        landscape_height = min(v.shape[0], v.shape[1])
        break

    height = landscape_height + landscape_height
    width = landscape_width + landscape_height + landscape_width
    tiled_im_bgr: NDArrayByte = np.zeros((height, width, 3), dtype=np.uint8)

    if "ring_front_left" in named_sensors:
        ring_front_left = named_sensors["ring_front_left"]
        tiled_im_bgr[:landscape_height, :landscape_width] = ring_front_left

    if "ring_front_center" in named_sensors:
        ring_front_center = named_sensors["ring_front_center"]
        tiled_im_bgr[
            :landscape_width, landscape_width : landscape_width + landscape_height
        ] = ring_front_center

    if "ring_front_right" in named_sensors:
        ring_front_right = named_sensors["ring_front_right"]
        tiled_im_bgr[:landscape_height, landscape_width + landscape_height :] = (
            ring_front_right
        )

    if "ring_side_left" in named_sensors:
        ring_side_left = named_sensors["ring_side_left"]
        tiled_im_bgr[landscape_height : 2 * landscape_height, :landscape_width] = (
            ring_side_left
        )

    if "ring_side_right" in named_sensors:
        ring_side_right = named_sensors["ring_side_right"]
        tiled_im_bgr[
            landscape_height : 2 * landscape_height,
            landscape_width + landscape_height :,
        ] = ring_side_right

    if bev_img is not None:
        tiled_im_bgr[
            landscape_width : 2 * landscape_width,
            landscape_width : landscape_width + landscape_height,
        ] = bev_img

    if "ring_rear_left" in named_sensors:
        ring_rear_left = named_sensors["ring_rear_left"]
        tiled_im_bgr[2 * landscape_height : 3 * landscape_height, :landscape_width] = (
            ring_rear_left
        )

    if "ring_rear_right" in named_sensors:
        ring_rear_right = named_sensors["ring_rear_right"]
        tiled_im_bgr[
            2 * landscape_height : 3 * landscape_height, width - landscape_width :
        ] = ring_rear_right
    return tiled_im_bgr

def preview_camera_captures(
    input_root: str = 'data/argoverse/processed/training',
    output_root: str = 'data/argoverse/preview/camera_captures',
    frame_width: int = int(5646 / 4), # Concatenate five camera images horizontally
    frame_height: int = int(3100 / 4),
    fps: int = 16
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
        num_frames = num_images // 6
        
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        for frame_id in range(num_frames):
            front = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 5]))
            front_left = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 5 + 1]))
            front_right = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 5 + 2]))
            left = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 5 + 3]))
            right = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 5 + 4]))
            
            combined_img = tile_cameras({
                "ring_front_left": front_left,
                "ring_front_center": front,
                "ring_front_right": front_right,
                "ring_side_left": left,
                "ring_side_right": right
            })
            combined_img = cv2.resize(combined_img, (int(frame_width), int(frame_height)))
            
            out.write(combined_img)
        out.release()

def preview_3dbox_projection(
    input_root: str = 'data/argoverse/processed/training',
    output_root: str = 'data/argoverse/preview/3dbox_vis',
    frame_width: int = int(5646 / 4),
    frame_height: int = int(3100 / 4),
    fps: int = 24
):
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # Iterate over each scene directory in the input root directory
    for scene_id in tqdm(sorted(os.listdir(input_root))):
        scene_folder = os.path.join(input_root, scene_id, "3dbox_vis")
        
        # Define the output video file path
        output_video_path = os.path.join(output_root, f'{scene_id}.mp4')
            
        # Get paths to each camera image folder
        images_all = sorted([f for f in sorted(os.listdir(scene_folder)) if f.lower().endswith('.jpg')])
        num_images = len(images_all)
        num_frames = num_images // 6
        
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        for frame_id in range(num_frames):
            front = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 5]))
            front_left = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 5 + 1]))
            front_right = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 5 + 2]))
            left = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 5 + 3]))
            right = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 5 + 4]))
            
            combined_img = tile_cameras({
                "ring_front_left": front_left,
                "ring_front_center": front,
                "ring_front_right": front_right,
                "ring_side_left": left,
                "ring_side_right": right
            })
            combined_img = cv2.resize(combined_img, (int(frame_width), int(frame_height)))
            
            out.write(combined_img)
        out.release()

def preview_dynamic_masks(
    input_root: str = 'data/argoverse/processed/training',
    output_root: str = 'data/argoverse/preview/dynamic_masks',
    frame_width: int = int(5646 / 4),
    frame_height: int = int(3100 / 4),
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
        num_frames = num_images // 6
        
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        for frame_id in range(num_frames):
            front = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 5]))
            front_left = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 5 + 1]))
            front_right = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 5 + 2]))
            left = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 5 + 3]))
            right = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 5 + 4]))
            
            combined_img = tile_cameras({
                "ring_front_left": front_left,
                "ring_front_center": front,
                "ring_front_right": front_right,
                "ring_side_left": left,
                "ring_side_right": right
            })
            combined_img = cv2.resize(combined_img, (int(frame_width), int(frame_height)))
            
            out.write(combined_img)
        out.release()
        
def preview_instances(
    input_root: str = 'data/argoverse/processed/training',
    output_root: str = 'data/argoverse/preview/instances',
    frame_width: int = int(5646 / 4),
    frame_height: int = int(3100 / 4),
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
        num_frames = num_images // 6
        
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        for frame_id in range(num_frames):
            front = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 5]))
            front_left = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 5 + 1]))
            front_right = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 5 + 2]))
            left = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 5 + 3]))
            right = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 5 + 4]))
            
            combined_img = tile_cameras({
                "ring_front_left": front_left,
                "ring_front_center": front,
                "ring_front_right": front_right,
                "ring_side_left": left,
                "ring_side_right": right
            })
            combined_img = cv2.resize(combined_img, (int(frame_width), int(frame_height)))
            
            out.write(combined_img)
        out.release()
        
def cat_all_videos(
    keys: list,
    output_root: str = 'data/argoverse/preview/catted_videos',
    scene_id: str = 'all',
    fps: int = 24
):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    if scene_id == 'all':
        scene_ids = sorted(os.listdir('data/argoverse/processed/training'))
    else:
        scene_ids = [scene_id]
        
    for scene_id in scene_ids:
        video_paths = []
        for key in keys:
            video_paths.append(f'data/argoverse/preview/{key}/{scene_id}.mp4')
        
        output_video_path = os.path.join(output_root, f'{scene_id}.mp4')

        # use moviepy to concatenate videos
        clips = [VideoFileClip(path) for path in video_paths]
        # cat videos from top to bottom, vertically
        final_clip = clips_array([[clip] for clip in clips])
        final_clip.write_videofile(output_video_path, fps=fps)
        final_clip.close()
            
if __name__ == '__main__':
    preview_camera_captures(fps=24)
    preview_3dbox_projection(fps=24)
    preview_dynamic_masks(fps=24)
    preview_instances(fps=24)
    cat_all_videos(keys=['camera_captures', '3dbox_vis', 'dynamic_masks', 'instances'], scene_id='all', fps=24)