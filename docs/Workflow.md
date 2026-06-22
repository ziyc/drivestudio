# DriveStudio Workflow (Waymo)

端到端工作流：环境搭建 → 下载预处理 → 批量训练 → 渲染导出 → 轨迹可视化。

---

## 0. 环境

```bash
# 安装依赖
uv sync
uv sync --group data          # 含 transformers（extract_masks 需要）
bash scripts/check_env.sh     # 验证环境

# SMPL 模型（一次性）
# 下载 SMPL v1.1，将 basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl 放到
# smpl_models/SMPL_NEUTRAL.pkl
```

---

## 1. 下载 Waymo 数据集

```bash
# 安装 Waymo toolkit
pip install waymo-open-dataset-tf-2-11-0==1.6.0

mkdir -p data/waymo/raw data/waymo/processed

# 维护自己的场景列表 data/waymo_my_scenes.txt（一行一个 scene ID）
# 已处理过的场景会自动跳过（检查 data/waymo/processed/training/{id}/）
python datasets/waymo/waymo_download.py \
    --target_dir data/waymo/raw \
    --scene_file data/waymo_my_scenes.txt
```

---

## 2. 预处理

```bash
# 主预处理
python datasets/preprocess.py \
    --data_root data/waymo/raw/ \
    --target_dir data/waymo/processed \
    --dataset waymo --split training \
    --scene_ids 1 2 3 4 5 6 7 8 9 10 23 114 172 327 552 621 703 788 \
    --workers 8 \
    --process_keys images lidar calib pose dynamic_masks objects

# 提取 fine dynamic mask
python datasets/tools/extract_masks.py \
    --data_root data/waymo/processed/training \
    --scene_ids 1 2 3 4 5 6 7 8 9 10 23 114 172 327 552 621 703 788 \
    --process_dynamic_mask
```

人姿态数据可以下载预处理好的：
```bash
# 下载预处理过的人姿态
cd data/
gdown 1QrtMrPAQhfSABpfgQWJZA2o_DDamL_7_
unzip waymo_preprocess_humanpose.zip && rm waymo_preprocess_humanpose.zip
cd ..
```

---

## 3. 批量训练

```bash
# 批量训练（自动发现场景、分配 GPU）
bash scripts/launch_train_batch.sh \
    --dataset waymo \
    --camera-sets 3cams,5cams \
    --gpus 0,1,2,3 \
    --scenes 1,2,3,4,5,6,7,8,9,10,23,114,172,327,552,621,703,788 \
    --load-smpl true

# 只训练缺失的场景
bash scripts/launch_train_batch.sh \
    --dataset waymo --camera-sets 3cams,5cams \
    --gpus 0,1,2,3 --only-missing

# 单场景训练
python tools/train.py --config_file configs/omnire.yaml \
    --output_root ./outputs \
    --project waymo --run_name scene023_5cams \
    dataset=waymo/5cams data.scene_idx=23
```

**训练产出目录结构：**
```
outputs/train/waymo/scene023/5cams_step30000_smpl/
    config.yaml
    checkpoint_final.pth       # 30000 步
    checkpoint_20000.pth
    checkpoint_10000.pth
    checkpoint_5000.pth
    ...
```

---

## 4. 渲染导出

### 4a. `tools/build_omnire_dataset.py` — 构建 OmniRe 标准数据集

根据 step checkpoint 生成不同 gaussian drop 等级的渲染视频，用于 Cosmos 后训练。

**输出布局：**
```
{output_root}/
    videos/          # GT（各模式共用）
    render/          # gaussian_drop=0.00 (checkpoint_final.pth)
    light/           # gaussian_drop=0.10 (checkpoint_20000.pth)
    medium/          # gaussian_drop=0.25 (checkpoint_10000.pth)
    heavy/           # gaussian_drop=0.40 (checkpoint_5000.pth)
```

**命令：**
```bash
python tools/build_omnire_dataset.py \
    --scene_ids 023 114 172 \
    --checkpoint_root "outputs/train/waymo/scene{scene}/5cams_step30000_smpl" \
    --output_root data/omnire/waymo

# 只渲染 clean render，跳过已有
python tools/build_omnire_dataset.py \
    --scene_ids 023 --modes render --skip_existing \
    --checkpoint_root "outputs/train/waymo/scene{scene}/5cams_step30000_smpl"

# 试运行（只打印计划）
python tools/build_omnire_dataset.py \
    --scene_ids 023 --dry-run \
    --checkpoint_root "outputs/train/waymo/scene{scene}/5cams_step30000_smpl"
```

### 4b. `tools/build_novel_view_split.py` — 生成新视角渲染

按 split 文件渲染 lane_offset 等 novel trajectory。

**命令：**
```bash
python tools/build_novel_view_split.py \
    --split_file data/waymo_test_split.txt \
    --checkpoint_root "outputs/train/waymo/scene{scene}/5cams_step30000_smpl" \
    --traj_types lane_offset_left lane_offset_right \
    --lane_offset_ratios 0.1 0.2 0.3

# dry run 查看计划
python tools/build_novel_view_split.py \
    --split_file data/waymo_test_split.txt \
    --checkpoint_root "outputs/train/waymo/scene{scene}/5cams_step30000_smpl" \
    --dry-run
```

---

## 5. 轨迹可视化

### 5a. `tools/visualize_trajectories.py` — 纯轨迹图

不需要 checkpoint，直接从位姿数据生成轨迹 3D 图。

```bash
python tools/visualize_trajectories.py \
    --config_file configs/omnire.yaml \
    dataset=waymo/5cams data.scene_idx=23 \
    --traj_types ego_raw lane_offset_left lane_offset_right \
    --lane_offset 3.5

# 输出: ./outputs/traj/waymo-s023-*/(trajectory.npy, trajectory_3d.png)
```

### 5b. `tools/render.py` — 单场景新轨迹渲染

从单个 checkpoint 渲染注册的 novel trajectory 并输出视频。

```bash
# 列出现有轨迹名称
python tools/render.py --list_traj_types

# 渲染 lane_offset 轨迹
python tools/render.py \
    --resume_from outputs/train/waymo/scene023/5cams_step30000_smpl/checkpoint_final.pth \
    --traj_types lane_offset_left lane_offset_right \
    --lane_offset_ratio 0.15

# 输出: ./outputs/render/<run_name>/(*.mp4, *_traj.png)
```

---

## 6. 评估

```bash
python tools/eval.py \
    --resume_from outputs/train/waymo/scene023/5cams_step30000_smpl/checkpoint_final.pth

# 输出: <log_dir>/metrics_eval/ 和 <log_dir>/videos_eval/
```

---

## 7. 服务端同步

服务器 `zkrh` 路径与本地一致。常用同步方式：

```bash
# 推送到服务器
rsync -avzP --exclude='checkpoint_*.pth' data/waymo/processed/ zkrh:~/Code/active/drivestudio/data/waymo/processed/
rsync -avzP outputs/ zkrh:~/Code/active/drivestudio/outputs/

# 从服务器拉取渲染结果
rsync -avzP zkrh:~/Code/active/drivestudio/outputs/datasets/ ./outputs/datasets/
rsync -avzP zkrh:~/Code/active/drivestudio/data/omnire/ ./data/omnire/

# 同步脚本（含渲染工具）
rsync -avzP tools/build_*.py zkrh:~/Code/active/drivestudio/tools/
```

**注意：** 服务端只有处理后数据，无 raw TFRecord；训练/渲染产出都在相同路径结构下。
