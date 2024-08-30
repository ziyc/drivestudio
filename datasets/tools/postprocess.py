from typing import List, Dict
import os
import joblib

import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

def interpolate_matrix(matrix1, matrix2, fraction):
    # use quaternion slerp
    q1 = matrix_to_quaternion(matrix1)
    q2 = matrix_to_quaternion(matrix2)
    dot = (q1 * q2).sum(dim=-1)
    dot = torch.clamp(dot, -1, 1)
    
    neg_mask = dot < 0
    q2[neg_mask] = -q2[neg_mask]
    dot[neg_mask] = -dot[neg_mask]
    
    similar_mask = dot > 0.9995
    q_interp_similar = q1 + fraction * (q2 - q1)
        
    theta_0 = torch.acos(dot)
    theta = theta_0 * fraction
    
    sin_theta = torch.sin(theta)
    sin_theta_0 = torch.sin(theta_0)
    
    s1 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0
    
    q_interp = (s1[:, None] * q1) + (s2[:, None] * q2)
    
    final_q_interp = torch.zeros_like(q1)
    final_q_interp[similar_mask] = q_interp_similar[similar_mask]
    final_q_interp[~similar_mask] = q_interp[~similar_mask]
    return quaternion_to_matrix(final_q_interp)

def compute_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = w1 * h1
    boxBArea = w2 * h2
    union_area = boxAArea + boxBArea - interArea
    iou = (interArea / union_area) if union_area > 0 else 0
    return iou

def interpolate_features(mask, features, is_rot_mat=False):
    features = features.clone()
    n = features.shape[0]
    for i in range(n):
        if not mask[i]:
            prev_index = i - 1
            next_index = i + 1
            
            while prev_index >= 0 and not mask[prev_index]:
                prev_index -= 1
                
            while next_index < n and not mask[next_index]:
                next_index += 1
                
            if prev_index >= 0 and next_index < n:
                interp_factor = (i - prev_index) / (next_index - prev_index)
                if not is_rot_mat:
                    features[i] = (1 - interp_factor) * features[prev_index] + interp_factor * features[next_index]
                else:
                    features[i] = interpolate_matrix(features[prev_index], features[next_index], interp_factor)
                
            elif prev_index >= 0:
                features[i] = features[prev_index]
            elif next_index < n:
                features[i] = features[next_index]
    return features

def detect_breaks_mask(bool_sequence):
    mask = [False] * len(bool_sequence)
    in_true_sequence = False
    start_false_sequence = False
    last_true_index = -1  # 跟踪最后一个True的位置

    for i, value in enumerate(bool_sequence):
        if value:
            if start_false_sequence:
                start_false_sequence = False
                if last_true_index != -1 and i < len(bool_sequence) - 1:
                    mask[last_true_index + 1:i] = [True] * (i - last_true_index - 1)
            in_true_sequence = True
            last_true_index = i
        else:
            if in_true_sequence:
                start_false_sequence = True
                in_true_sequence = False

    if start_false_sequence and last_true_index < len(bool_sequence) - 1:
        mask[last_true_index + 1:len(bool_sequence)] = [False] * (len(bool_sequence) - last_true_index - 1)

    return mask
    
def match_and_postprocess(
    scene_dir: str,
    GTTracksDict: Dict[int, Dict],
    PredTracksDict: Dict[int, Dict],
    camera_list: List[int],
    save_temp=True,
    verbose=False,
    fps=12
):
    """Match tha predicted tracks to the GT tracks, and postprocess the matched 
    tracks: complete the missing frames, interpolate the missing values, etc.
    
    Args:
        scene_dir: str, the path to the scene directory
        GTTracksDict: Dict[int, Dict], the dictionary of GT tracks
        PredTracksDict: Dict[int, Dict], the dictionary of predicted tracks
        camera_list: List[int], the list of camera IDs
        save_temp: bool, whether to save the temporary results
        verbose: bool, whether to print the verbose information
        fps: int, the frame rate of the video, if save the video, the fps will be used
        
    Returns:
        merged_collector: Dict[int, Dict], the dictionary of proprocessed SMPL parameters
    """
    # number of cameras
    num_c = len(camera_list)
    
    if save_temp:
        temp_dir = os.path.join(scene_dir, "humanpose", "postprocess")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
    
    # --------------------------------------------------------------------------
    #                 Parse ground truth and predicted tracks
    # --------------------------------------------------------------------------
    parsed_pred_track_all, parsed_gt_track_all = {}, {}
    for cam_id in camera_list:
        parsed_pred_track, parsed_gt_track = {}, {}

        raw_pred_track = PredTracksDict[cam_id]
        raw_gt_track = GTTracksDict[cam_id]
        assert len(raw_pred_track) == len(raw_gt_track), \
            f"len(raw_pred_track) {len(raw_pred_track)} != len(raw_gt_track) {len(raw_gt_track)}"
        num_f = len(raw_pred_track) # number of frames
        
        for fi, pred_track_f in enumerate(raw_pred_track.values()):
            for i_, tid in enumerate(pred_track_f["tid"]):
                if pred_track_f["tracked_time"][i_] != 0:
                    continue
                if tid not in parsed_pred_track:
                    parsed_pred_track[tid] = {
                        "valids": torch.zeros(num_f, dtype=torch.bool),
                        "tracked_bbox": torch.zeros(num_f, 4),
                        "smpl": {
                            "global_orient": torch.zeros(num_f, 1, 3, 3),
                            "body_pose": torch.zeros(num_f, 23, 3, 3),
                            "betas": torch.zeros(num_f, 10),
                        },
                        "camera": torch.zeros(num_f, 3),
                    }
                parsed_pred_track[tid]["valids"][fi] = True
                parsed_pred_track[tid]["tracked_bbox"][fi] = torch.from_numpy(pred_track_f["bbox"][i_])
                parsed_pred_track[tid]["smpl"]["global_orient"][fi] = torch.tensor(pred_track_f["smpl"][i_]["global_orient"])
                parsed_pred_track[tid]["smpl"]["body_pose"][fi] = torch.tensor(pred_track_f["smpl"][i_]["body_pose"])
                parsed_pred_track[tid]["smpl"]["betas"][fi] = torch.tensor(pred_track_f["smpl"][i_]["betas"])
                parsed_pred_track[tid]["camera"][fi] = torch.tensor(pred_track_f["camera"][i_])
        
        # collect GT track ids
        for fi, gt_track_f in enumerate(raw_gt_track.values()):
            for i_, tid in enumerate(gt_track_f["extra_data"]["gt_track_id"]):
                if tid not in parsed_gt_track:
                    parsed_gt_track[tid] = {
                        "valids": torch.zeros(num_f, dtype=torch.bool),
                        "tracked_bbox": torch.zeros(num_f, 4),
                    }
                parsed_gt_track[tid]["valids"][fi] = True
                parsed_gt_track[tid]["tracked_bbox"][fi] = torch.tensor(gt_track_f["gt_bbox"][i_])
        
        parsed_pred_track_all[cam_id] = parsed_pred_track
        parsed_gt_track_all[cam_id] = parsed_gt_track
    
    # --------------------------------------------------------------------------
    #             Find GT tracks that have at least one 4D-Humans prediction
    # --------------------------------------------------------------------------
    # For each camera:
    #   1. Compare predicted tracks to GT tracks using IoU (Intersection over Union)
    #   2. Link each predicted track to its best-matching GT track
    #
    # Key points:
    # - Some GT tracks may not have predictions (e.g., far-away pedestrians)
    # - A GT track might match multiple predicted tracks
    # - We keep only GT tracks that have at least one prediction in any camera
    #
    # Result: valid_gt_tids list contains GT tracks visible to 4D-Humans in at least one camera
    valid_gt_tids = []
    match_pred2gt = {}
    for cam_id in camera_list: 
        parsed_pred_track = parsed_pred_track_all[cam_id]
        parsed_gt_track = parsed_gt_track_all[cam_id]
        _matches = {}
        for pred_tid in parsed_pred_track.keys():
            max_iou = 0
            matched_gt_tid = -1
            for gt_tid in parsed_gt_track.keys():
                iou = 0
                for fi in range(num_f):
                    if parsed_pred_track[pred_tid]["valids"][fi] and parsed_gt_track[gt_tid]["valids"][fi]:
                        iou += compute_iou(parsed_pred_track[pred_tid]["tracked_bbox"][fi], parsed_gt_track[gt_tid]["tracked_bbox"][fi])
                iou /= num_f
                if iou > max_iou:
                    max_iou = iou
                    matched_gt_tid = gt_tid
            if matched_gt_tid == -1:
                continue
            _matches[pred_tid] = matched_gt_tid
        
        print(f"cam_id {cam_id}: {len(_matches)} pred tracks matched")
        print("_matches:", _matches)
            
        valid_gt_tids += list(_matches.values())
        match_pred2gt[cam_id] = _matches

    # get unique valid GT IDs
    valid_gt_tids = list(set(valid_gt_tids))
    print(f"valid_gt_tids: {valid_gt_tids}")
    
    # --------------------------------------------------------------------------
    #                   Link GT tracks to predicted tracks
    # --------------------------------------------------------------------------
    # For each camera:
    #   1. Calculate average IoU between GT and predicted tracks
    #   2. Match each GT track to its best predicted track (if any)
    #
    # Key points:
    # - A GT track can match different predicted tracks across cameras
    # - The same GT Track may have different predicted track IDs across cameras
    #   due to independent processing of each camera's video
    # - It connects the consistent GT ID to camera-specific predicted IDs
    #
    # Result: We create a map from GT tracks to their best predicted tracks (if any) in each camera
    match_gt2pred = {}
    for cam_id in camera_list:
        parsed_pred_track = parsed_pred_track_all[cam_id]
        parsed_gt_track = parsed_gt_track_all[cam_id]
        _matches = {}
        for gt_tid in valid_gt_tids:
            if gt_tid not in parsed_gt_track:
                continue
            max_iou = 0
            matched_pred_tid = -1
            for pred_tid in parsed_pred_track.keys():
                iou = 0
                for fi in range(num_f):
                    if parsed_pred_track[pred_tid]["valids"][fi] and parsed_gt_track[gt_tid]["valids"][fi]:
                        iou += compute_iou(parsed_pred_track[pred_tid]["tracked_bbox"][fi], parsed_gt_track[gt_tid]["tracked_bbox"][fi])
                iou /= num_f
                if iou > max_iou:
                    max_iou = iou
                    matched_pred_tid = pred_tid
                _matches[gt_tid] = matched_pred_tid
        
        print(f"cam_id {cam_id}: {len(_matches)} GT tracks matched")
        print("_matches:", _matches)
        match_gt2pred[cam_id] = _matches
        
    # --------------------------------------------------------------------------
    #           Collect SMPL parameters for each valid GT track
    # --------------------------------------------------------------------------
    # This step processes the results of GT-to-pred track matching across multiple cameras:
    # 1. For each valid GT track ID:
    #    a. Initialize data structures to store SMPL parameters, camera info, and validity masks
    #    b. For each camera:
    #       - Record when the GT track appears in 2D (2DBox_appear_mask)
    #       - If matched to a pred track:
    #         * Collect SMPL parameters (global_orient, body_pose, betas) from the matched pred track
    #         * Collect camera parameters
    #         * Mark frames where data is available (matched_mask)
    # 2. The collected data retains the multi-camera structure, allowing for later analysis
    #    of track consistency across different views
    #
    # This approach allows us to:
    # - Maintain the GT track ID as the primary identifier
    # - Collect all available pose data for each GT track from matched predictions
    # - Preserve the multi-camera nature of the data for subsequent processing steps
    
    collector = {}
    # Initialize the collector
    for gt_tid in valid_gt_tids:
        collector[gt_tid] = {
            "smpl": {
                "global_orient": torch.zeros(num_c, num_f, 1, 3, 3),
                "body_pose": torch.zeros(num_c, num_f, 23, 3, 3),
                "betas": torch.zeros(num_c, num_f, 10),
            },
            "camera": torch.zeros(num_c, num_f, 3),
            "2DBox_appear_mask": torch.zeros(num_c, num_f).bool(),
            "area": torch.zeros(num_c, num_f),
            "matched_mask": torch.zeros(num_c, num_f).bool(),
        }
        
    # collect data
    for cam_id in camera_list:
        parsed_pred_track = parsed_pred_track_all[cam_id]
        parsed_gt_track = parsed_gt_track_all[cam_id]
        _matches = match_gt2pred[cam_id]
        
        if len(_matches) > 0:
            # collect 2DBox_appear_mask
            for gt_tid in valid_gt_tids:
                if gt_tid not in parsed_gt_track:
                    continue
                for fi in range(num_f):
                    if parsed_gt_track[gt_tid]["valids"][fi]:
                        collector[gt_tid]["2DBox_appear_mask"][cam_id, fi] = 1
                        box = parsed_gt_track[gt_tid]["tracked_bbox"][fi]
                        collector[gt_tid]["area"][cam_id, fi] = box[2] * box[3]
            
            # collect SMPL data
            for gt_tid in valid_gt_tids:
                if gt_tid not in parsed_gt_track:
                    continue
                matched_pred_tid = _matches[gt_tid]
                if matched_pred_tid == -1:
                    continue
                for fi in range(num_f):
                    if parsed_gt_track[gt_tid]["valids"][fi] and parsed_pred_track[matched_pred_tid]["valids"][fi]:
                        collector[gt_tid]["smpl"]["global_orient"][cam_id, fi] = parsed_pred_track[matched_pred_tid]["smpl"]["global_orient"][fi].clone()
                        collector[gt_tid]["smpl"]["body_pose"][cam_id, fi] = parsed_pred_track[matched_pred_tid]["smpl"]["body_pose"][fi].clone()
                        collector[gt_tid]["smpl"]["betas"][cam_id, fi] = parsed_pred_track[matched_pred_tid]["smpl"]["betas"][fi].clone()
                        collector[gt_tid]["camera"][cam_id, fi] = parsed_pred_track[matched_pred_tid]["camera"][fi].clone()
                        collector[gt_tid]["matched_mask"][cam_id, fi] = 1
        
    # update valid mask
    for gt_tid in valid_gt_tids:
        appear_mask = collector[gt_tid]["2DBox_appear_mask"]
        matched_mask = collector[gt_tid]["matched_mask"]
        # valid mask: 2DBox_appear_mask & matched_mask
        valid_mask = appear_mask & matched_mask
        collector[gt_tid]["valid_mask"] = valid_mask
    
    # Save matched data in a format compatible with 4D-Humans visualization
    if save_temp:
        for cam_id in camera_list:
            pkl_dict = {}
            for fi in range(num_f):
                fi_info = {
                    "time": fi,
                    "shot": 0,
                    "tid": [],
                    "tracked_time": [],
                    "mask": None,
                    "bbox": None,
                    "smpl": [],
                    "camera": [],
                }
                for gt_tid in valid_gt_tids:
                    if collector[gt_tid]["valid_mask"][cam_id, fi]:
                        fi_info["tid"].append(gt_tid)
                        fi_info["tracked_time"].append(0)
                        fi_info["smpl"].append({
                            "global_orient": collector[gt_tid]["smpl"]["global_orient"][cam_id, fi].numpy(),
                            "body_pose": collector[gt_tid]["smpl"]["body_pose"][cam_id, fi].numpy(),
                            "betas": collector[gt_tid]["smpl"]["betas"][cam_id, fi].numpy(),
                        })
                        fi_info["camera"].append(collector[gt_tid]["camera"][cam_id, fi].numpy())
                pkl_dict[fi] = fi_info
            joblib.dump(
                pkl_dict, os.path.join(temp_dir, f"{cam_id}_matched.pkl")
            )
    
    # --------------------------------------------------------------------------
    #           Complete data for all frames where GT 2D boxes exist
    # --------------------------------------------------------------------------
    # For each camera and GT track:
    #   1. Check each frame where a GT 2D box exists (human is visible in scene)
    #   2. If no matched data exists for that frame:
    #      - This often occurs due to occlusion or tracking failures
    #      - We have the 2D box (know human's position) but lack SMPL params
    #   3. Interpolate missing data (SMPL params, camera info) from surrounding frames
    #   4. This ensures continuity in human pose data even when direct detection fails
    #   5. Result: All frames with 2DBox_appear_mask==True will have complete data
    # 
    # This step is crucial for handling temporary occlusions or detection misses,
    # maintaining a consistent track even when the pose predictor fails momentarily.
    for cam_id in camera_list:
        for gt_tid in valid_gt_tids:
            need_fill = collector[gt_tid]["2DBox_appear_mask"][cam_id] & (~collector[gt_tid]["matched_mask"][cam_id])
            if need_fill.any() and not collector[gt_tid]["matched_mask"][cam_id].sum() == 0:
                appear_mask = collector[gt_tid]["2DBox_appear_mask"][cam_id]
                mask = collector[gt_tid]["valid_mask"][cam_id][appear_mask]
                collector[gt_tid]["matched_mask"][cam_id] = collector[gt_tid]["matched_mask"][cam_id] | appear_mask
                
                feature_masked = collector[gt_tid]["smpl"]["global_orient"][cam_id, appear_mask]
                feature_full = interpolate_features(mask, feature_masked, is_rot_mat=True)
                collector[gt_tid]["smpl"]["global_orient"][cam_id, appear_mask] = feature_full
                
                feature_masked = collector[gt_tid]["smpl"]["body_pose"][cam_id, appear_mask]
                feature_full = interpolate_features(mask, feature_masked, is_rot_mat=True)
                collector[gt_tid]["smpl"]["body_pose"][cam_id, appear_mask] = feature_full
                
                feature_masked = collector[gt_tid]["smpl"]["betas"][cam_id, appear_mask]
                feature_full = interpolate_features(mask, feature_masked)
                collector[gt_tid]["smpl"]["betas"][cam_id, appear_mask] = feature_full
                
                feature_masked = collector[gt_tid]["camera"][cam_id, appear_mask]
                feature_full = interpolate_features(mask, feature_masked)
                collector[gt_tid]["camera"][cam_id, appear_mask] = feature_full

    # Save completed data in a format compatible with 4D-Humans visualization
    if save_temp:
        for cam_id in camera_list:
            pkl_dict = {}
            for fi in range(num_f):
                fi_info = {
                    "time": fi,
                    "shot": 0,
                    "tid": [],
                    "tracked_time": [],
                    "mask": None,
                    "bbox": None,
                    "smpl": [],
                    "camera": [],
                }
                for gt_tid in valid_gt_tids:
                    if collector[gt_tid]["2DBox_appear_mask"][cam_id, fi]: # the difference is we use 2DBox_appear_mask here
                        fi_info["tid"].append(gt_tid)
                        fi_info["tracked_time"].append(0)
                        fi_info["smpl"].append({
                            "global_orient": collector[gt_tid]["smpl"]["global_orient"][cam_id, fi].numpy(),
                            "body_pose": collector[gt_tid]["smpl"]["body_pose"][cam_id, fi].numpy(),
                            "betas": collector[gt_tid]["smpl"]["betas"][cam_id, fi].numpy(),
                        })
                        fi_info["camera"].append(collector[gt_tid]["camera"][cam_id, fi].numpy())
                pkl_dict[fi] = fi_info
            joblib.dump(
                pkl_dict, os.path.join(temp_dir, f"{cam_id}_completed.pkl")
            )
    
    # --------------------------------------------------------------------------
    #              Merge and refine SMPL data across all cameras
    # --------------------------------------------------------------------------
    # 1. Initialize merged_collector:
    #    - For each GT track, create a structure to hold combined data from all cameras
    # 2. Merge data for each GT track:
    #    a. Combine matched_mask from all cameras
    #    b. Select best SMPL parameters for each frame:
    #       - If only one camera detects the person, use that camera's data
    #       - If multiple cameras detect (e.g., person moving across camera views):
    #         * Choose the camera with largest detection area
    #         * This typically provides the most accurate pose estimation
    #    c. Set must_appear_mask where the person is visible in any camera
    #    d. Store selected SMPL parameters, camera info, and 2D box appearance masks
    merged_collector = {}
    for gt_tid in valid_gt_tids:
        merged_collector[gt_tid] = {
            "smpl": {
                "global_orient": torch.zeros(num_f, 1, 3, 3),
                "body_pose": torch.zeros(num_f, 23, 3, 3),
                "betas": torch.zeros(num_f, 10),
            },
            "camera": torch.zeros(3, num_f, 3), # 3 cameras
            "2DBox_appear_mask": torch.zeros(3, num_f).bool(),
            "matched_mask": torch.zeros(num_f).bool(), # matched_mask for all cameras
            "must_appear_mask": torch.zeros(num_f).bool(),
            "selected_cam_idx": torch.zeros(num_f).long() - 1,  # -1 means not selected
        }
        matched_sum = collector[gt_tid]["matched_mask"].sum(0)
        merged_collector[gt_tid]["matched_mask"] = matched_sum > 0
        
        # if matched_sum == 1 means only one cam get the smpl, we just select the smpl from that cam
        matched_cam_id = collector[gt_tid]["matched_mask"].float().argmax(dim=0) 
        # if matched_sum > 1 means multiple cams get the smpl, we select the smpl from the cam with the largest area
        area = collector[gt_tid]["area"]
        max_cam_id = area.argmax(0)
        best_cam_id = matched_cam_id * (matched_sum == 1)+ max_cam_id * (matched_sum > 1)
        
        mask = collector[gt_tid]["2DBox_appear_mask"].sum(0) > 0
        merged_collector[gt_tid]["must_appear_mask"] = mask
        masked_best_cam_id = best_cam_id[mask]
        
        merged_collector[gt_tid]["smpl"]["global_orient"][mask] = collector[gt_tid]["smpl"]["global_orient"][:, mask][masked_best_cam_id, torch.arange(masked_best_cam_id.shape[0])]
        merged_collector[gt_tid]["smpl"]["body_pose"][mask] = collector[gt_tid]["smpl"]["body_pose"][:, mask][masked_best_cam_id, torch.arange(masked_best_cam_id.shape[0])]
        merged_collector[gt_tid]["smpl"]["betas"][mask] = collector[gt_tid]["smpl"]["betas"][:, mask][masked_best_cam_id, torch.arange(masked_best_cam_id.shape[0])]
        merged_collector[gt_tid]["camera"] = collector[gt_tid]["camera"]
        merged_collector[gt_tid]["2DBox_appear_mask"] = collector[gt_tid]["2DBox_appear_mask"]
        merged_collector[gt_tid]["selected_cam_idx"][mask] = masked_best_cam_id

    # 3. Fill gaps in SMPL data:
    #    - For frames where must_appear_mask is true but matched_mask is false:
    #      * Interpolate SMPL parameters (global_orient, body_pose, betas)
    #      * Interpolate camera info and selected camera index
    #    - This addresses frames where the person is visible but not detected
    for gt_tid in valid_gt_tids:
        need_fill = merged_collector[gt_tid]["must_appear_mask"] & (~merged_collector[gt_tid]["matched_mask"])
        if need_fill.any():
            final_complete_mask = merged_collector[gt_tid]["must_appear_mask"]
            mask = merged_collector[gt_tid]["matched_mask"][final_complete_mask]
            
            feature_masked = merged_collector[gt_tid]["smpl"]["global_orient"][final_complete_mask]
            feature_full = interpolate_features(mask, feature_masked, is_rot_mat=True)
            merged_collector[gt_tid]["smpl"]["global_orient"][final_complete_mask] = feature_full
            
            feature_masked = merged_collector[gt_tid]["smpl"]["body_pose"][final_complete_mask]
            feature_full = interpolate_features(mask, feature_masked, is_rot_mat=True)
            merged_collector[gt_tid]["smpl"]["body_pose"][final_complete_mask] = feature_full
            
            feature_masked = merged_collector[gt_tid]["smpl"]["betas"][final_complete_mask]
            feature_full = interpolate_features(mask, feature_masked)
            merged_collector[gt_tid]["smpl"]["betas"][final_complete_mask] = feature_full
            
            feature_masked = merged_collector[gt_tid]["camera"][:, final_complete_mask]
            feature_full = interpolate_features(mask, feature_masked)
            merged_collector[gt_tid]["camera"][:, final_complete_mask] = feature_full
            
            feature_masked = merged_collector[gt_tid]["selected_cam_idx"][final_complete_mask]
            feature_full = interpolate_features(mask, feature_masked)
            merged_collector[gt_tid]["selected_cam_idx"][final_complete_mask] = feature_full.round()
        
    # 4. Fill breaks in visibility:
    #    - Detect short gaps in must_appear_mask sequence
    #    - For these gaps:
    #      * Set must_appear_mask to true
    #      * Interpolate SMPL parameters, camera info, and selected camera index
    #    - This creates continuity in tracks, filling brief disappearances
    for gt_tid in valid_gt_tids:
        need_fill = torch.tensor(
            detect_breaks_mask(merged_collector[gt_tid]["must_appear_mask"])
        )
        if need_fill.any():
            final_complete_mask = need_fill | merged_collector[gt_tid]["must_appear_mask"]
            mask = merged_collector[gt_tid]["must_appear_mask"][final_complete_mask]
            
            feature_masked = merged_collector[gt_tid]["smpl"]["global_orient"][final_complete_mask]
            feature_full = interpolate_features(mask, feature_masked, is_rot_mat=True)
            merged_collector[gt_tid]["smpl"]["global_orient"][final_complete_mask] = feature_full
            
            feature_masked = merged_collector[gt_tid]["smpl"]["body_pose"][final_complete_mask]
            feature_full = interpolate_features(mask, feature_masked, is_rot_mat=True)
            merged_collector[gt_tid]["smpl"]["body_pose"][final_complete_mask] = feature_full
            
            feature_masked = merged_collector[gt_tid]["smpl"]["betas"][final_complete_mask]
            feature_full = interpolate_features(mask, feature_masked)
            merged_collector[gt_tid]["smpl"]["betas"][final_complete_mask] = feature_full
            
            feature_masked = merged_collector[gt_tid]["camera"][:, final_complete_mask]
            feature_full = interpolate_features(mask, feature_masked)
            merged_collector[gt_tid]["camera"][:, final_complete_mask] = feature_full
            
            feature_masked = merged_collector[gt_tid]["selected_cam_idx"][final_complete_mask]
            feature_full = interpolate_features(mask, feature_masked)
            merged_collector[gt_tid]["selected_cam_idx"][final_complete_mask] = feature_full.round()
            
            merged_collector[gt_tid]["must_appear_mask"] = final_complete_mask
    
    # update valid_mask
    for gt_tid in valid_gt_tids:
        merged_collector[gt_tid]["valid_mask"] = merged_collector[gt_tid]["must_appear_mask"].clone()
    
    # Save merged data in a format compatible with 4D-Humans visualization
    if save_temp:
        pkl_dict = {}
        for cam_id in camera_list:
            for fi in range(num_f):
                fi_info = {
                    "time": fi,
                    "shot": 0,
                    "tid": [],
                    "tracked_time": [],
                    "mask": None,
                    "bbox": None,
                    "smpl": [],
                    "camera": [],
                }
                for gt_tid in valid_gt_tids:
                    if merged_collector[gt_tid]["2DBox_appear_mask"][cam_id, fi]:
                        fi_info["tid"].append(gt_tid)
                        fi_info["tracked_time"].append(0)
                        fi_info["smpl"].append({
                            "global_orient": merged_collector[gt_tid]["smpl"]["global_orient"][fi].numpy(),
                            "body_pose": merged_collector[gt_tid]["smpl"]["body_pose"][fi].numpy(),
                            "betas": merged_collector[gt_tid]["smpl"]["betas"][fi].numpy(),
                        })
                        fi_info["camera"].append(merged_collector[gt_tid]["camera"][cam_id, fi].numpy())
                pkl_dict[fi] = fi_info
            joblib.dump(
                pkl_dict, os.path.join(temp_dir, f"{cam_id}_merged.pkl")
            )
    
    # pop out 2DBox_appear_mask and selected_cam_idx and camera
    for gt_tid in valid_gt_tids:
        merged_collector[gt_tid].pop("2DBox_appear_mask")
        merged_collector[gt_tid].pop("camera")
        
    # check if nan or inf in smpl, for debug use
    for gt_tid in valid_gt_tids:
        for k, v in merged_collector[gt_tid]["smpl"].items():
            if v.isnan().any() or v.isinf().any():
                raise ValueError(f"gt_tid {gt_tid} has nan or inf in {k}")
    
    return merged_collector