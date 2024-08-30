import os
import pickle

import cv2
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip, clips_array
from tqdm import tqdm

def count_labels(
    base_path: str = 'data/pandaset/raw'
):
    combined_label_counts = pd.Series(dtype=int)

    # Traverse all scene directories
    for scene_dir in os.listdir(base_path):
        scene_path = os.path.join(base_path, scene_dir, 'annotations', 'cuboids')
        if os.path.isdir(scene_path):
            # Traverse all PKL files in each scene
            for pkl_file in os.listdir(scene_path):
                if pkl_file.endswith('.pkl'):
                    pkl_file_path = os.path.join(scene_path, pkl_file)
                    # Read the PKL file
                    with open(pkl_file_path, 'rb') as f:
                        data = pickle.load(f)
                        # Count labels in the current PKL file
                        label_counts = data['label'].value_counts()
                        # Merge current label counts into the total counts
                        combined_label_counts = combined_label_counts.add(label_counts, fill_value=0)
    
    total_label_counts_df = combined_label_counts.reset_index()
    total_label_counts_df.columns = ['Label', 'Count']

    print(total_label_counts_df)

    unique_labels = total_label_counts_df['Label'].tolist()
    
    return unique_labels

def preview_camera_captures(
    input_root: str = 'data/pandaset/processed',
    output_root: str = 'data/pandaset/preview/camera_captures',
    frame_width: int = 960 * 5, # Concatenate five camera images horizontally
    frame_height: int = 540,
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
            front = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 6]))
            front_left = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 6 + 1]))
            front_right = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 6 + 2]))
            left = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 6 + 3]))
            right = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 6 + 4]))
            
            combined_img = np.hstack((left, front_left, front, front_right, right))
            combined_img = cv2.resize(combined_img, (frame_width, frame_height))
            
            out.write(combined_img)
        out.release()

def preview_3dbox_projection(
    input_root: str = 'data/pandaset/processed',
    output_root: str = 'data/pandaset/preview/3dbox_vis',
    frame_width: int = 960 * 5,
    frame_height: int = 540,
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
            front = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 6]))
            front_left = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 6 + 1]))
            front_right = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 6 + 2]))
            left = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 6 + 3]))
            right = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 6 + 4]))
            
            combined_img = np.hstack((left, front_left, front, front_right, right))
            combined_img = cv2.resize(combined_img, (frame_width, frame_height))
            
            out.write(combined_img)
        out.release()

def preview_dynamic_masks(
    input_root: str = 'data/pandaset/processed',
    output_root: str = 'data/pandaset/preview/dynamic_masks',
    frame_width: int = 960 * 5,
    frame_height: int = 540,
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
            front = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 6]))
            front_left = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 6 + 1]))
            front_right = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 6 + 2]))
            left = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 6 + 3]))
            right = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 6 + 4]))
            
            combined_img = np.hstack((left, front_left, front, front_right, right))
            combined_img = cv2.resize(combined_img, (frame_width, frame_height))
            
            out.write(combined_img)
        out.release()
        
def preview_instances(
    input_root: str = 'data/pandaset/processed',
    output_root: str = 'data/pandaset/preview/instances',
    frame_width: int = 960 * 5,
    frame_height: int = 540,
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
            front = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 6]))
            front_left = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 6 + 1]))
            front_right = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 6 + 2]))
            left = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 6 + 3]))
            right = cv2.imread(os.path.join(scene_folder, images_all[frame_id * 6 + 4]))
            
            combined_img = np.hstack((left, front_left, front, front_right, right))
            combined_img = cv2.resize(combined_img, (frame_width, frame_height))
            
            out.write(combined_img)
        out.release()
        
def cat_all_videos(
    keys: list,
    output_root: str = 'data/pandaset/preview/catted_videos',
    scene_id: str = 'all',
    fps: int = 24
):
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    if scene_id == 'all':
        scene_ids = sorted(os.listdir('data/pandaset/processed'))
    else:
        scene_ids = [scene_id]
        
    for scene_id in scene_ids:
        video_paths = []
        for key in keys:
            video_paths.append(f'data/pandaset/preview/{key}/{scene_id}.mp4')
        
        output_video_path = os.path.join(output_root, f'{scene_id}.mp4')

        # use moviepy to concatenate videos
        clips = [VideoFileClip(path) for path in video_paths]
        # cat videos from top to bottom, vertically
        final_clip = clips_array([[clip] for clip in clips])
        final_clip.write_videofile(output_video_path, fps=fps)
        final_clip.close()
            
if __name__ == '__main__':
    # unique_labels = count_labels()
    preview_camera_captures(fps=16)
    preview_3dbox_projection(fps=16)
    preview_dynamic_masks(fps=16)
    preview_instances(fps=16)
    cat_all_videos(keys=['camera_captures', '3dbox_vis', 'dynamic_masks', 'instances'], scene_id='all', fps=16)