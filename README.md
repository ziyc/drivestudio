<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/media/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/media/logo.png">
    <img alt="Logo" src="docs/media/logo_clipped.png" width="700">
  </picture>
</p>
<p align="center">
A 3DGS framework for omni urban scene reconstruction and simulation!
</p>

<p align="center">
    <!-- project -->
    <a href="https://ziyc.github.io/omnire/"><img src="https://img.shields.io/badge/Project-Page-FFFACD" height="28"/></a>
    <!-- paper -->
    <a href="https://arxiv.org/abs/2408.16760">
        <img src='https://img.shields.io/badge/arXiv-Paper-E6E6FA' height="28"/>
    </a>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/08e6c613-f61a-4d0d-a2a9-1538fcd4f5ff" width="49%" style="max-width: 100%; height: auto;" />
  <img src="https://github.com/user-attachments/assets/d2a47e7d-2934-46de-94d6-85ea8a52aba6" width="49%" style="max-width: 100%; height: auto;" />
</p>

## About
DriveStudio is a 3DGS codebase for urban scene reconstruction/simulation. It offers a system with multiple Gaussian representations to jointly reconstruct backgrounds, vehicles, and non-rigid categories (pedestrians, cyclists, etc.) from driving logs. DriveStudio also provides a unified data system supporting various popular driving datasets, including [Waymo](https://waymo.com/open/), [PandaSet](https://pandaset.org/), [Argoverse2](https://www.argoverse.org/av2.html), [KITTI](http://www.cvlibs.net/datasets/kitti/), [NuScenes](https://www.nuscenes.org/), and [NuPlan](https://www.nuscenes.org/nuplan).

This codebase also contains the **official implementation** of:
  > **OmniRe: Omni Urban Scene Reconstruction** <br> ICLR 2025 (Spotlight) <br> [Project Page](https://ziyc.github.io/omnire/) | [Paper](https://arxiv.org/abs/2408.16760) <br> [Ziyu Chen](https://ziyc.github.io/), [Jiawei Yang](https://jiawei-yang.github.io/), [Jiahui Huang](https://huangjh-pub.github.io/), [Riccardo de Lutio](https://riccardodelutio.github.io/), [Janick Martinez Esturo](https://www.jme.pub/), [Boris Ivanovic](https://www.borisivanovic.com/), [Or Litany](https://orlitany.github.io/), [Zan Gojcic](https://zgojcic.github.io/), [Sanja Fidler](https://www.cs.utoronto.ca/~fidler/), [Marco Pavone](https://stanford.edu/~pavone/), [Li Song](https://medialab.sjtu.edu.cn/author/li-song/), [Yue Wang](https://yuewang.xyz/)

## 🎉 Try your own projects/research on DriveStudio!
### 🔥 Highlighted implementations

Our codebase supports two types of Gaussian trainers:

1. Single-Representation trainer (single Gaussian representation for the entire scene):
   - Deformable Gaussians
   - Periodic Vibration Gaussians

2. Multi-Representation trainer (Gaussian scene graphs trainer):
   - Background: Static Gaussians (Vanilla Gaussians)
   - Vehicles: Static Gaussians
   - Humans: SMPL-Gaussians, Deformable Gaussians
   - Other non-rigid categories: Deformable Gaussians

**Implemented methods:**

| Method Name | Implementation | Trainer Type | Gaussian Representations |
|-------------|----------------|--------------|--------------------------|
| [OmniRe](https://ziyc.github.io/omnire/) | Official | Multi | • Static Gaussians: Background, Vehicles<br>• SMPL Gaussians: Pedestrians (majority)<br>• Deformable Gaussians: Cyclists, far-range pedestrians, other non-rigid categories |
| [Deformable-GS](https://github.com/ingra14m/Deformable-3D-Gaussians) | Unofficial | Single | • Deformable Gaussians: Entire scene |
| [PVG](https://github.com/fudan-zvg/PVG) | Unofficial | Single | • Periodic Vibration Gaussians: Entire scene |
| [Street Gaussians](https://github.com/zju3dv/street_gaussians) | Unofficial | Multi | • Static Gaussians: Background, Vehicles |

We extend our gratitude to the authors for their remarkable contributions. If you find these works useful, please consider citing them.

### 🚗 Dataset Support
This codebase provides support for popular driving datasets. We offer instructions and scripts on how to download and process these datasets:

| Dataset | Instruction | Cameras | Sync Frequency | Object Annotation |
|---------|-------------|---------|----------------|-------------------|
| Waymo | [Data Process Instruction](docs/Waymo.md) | 5 cameras | 10Hz | ✅ |
| NuScenes | [Data Process Instruction](docs/NuScenes.md) | 6 cameras | 2Hz (up to 10Hz*) | ✅ |
| NuPlan | [Data Process Instruction](docs/Nuplan.md) | 8 cameras | 10Hz | ✅ |
| ArgoVerse | [Data Process Instruction](docs/ArgoVerse.md) | 7 cameras | 10Hz | ✅ |
| PandaSet | [Data Process Instruction](docs/Pandaset.md) | 6 cameras | 10Hz | ✅ |
| KITTI | [Data Process Instruction](docs/KITTI.md) | 2 cameras | 10Hz | ✅ |

*NOTE: For NuScenes data, LiDAR operates at 20Hz and cameras at 12Hz, but keyframes (with object annotations) are only at 2Hz. We provide a method to interpolate annotations up to 10Hz.

### ✨ Functionality

<details>
<summary>Click to expand functionality details</summary>

We have implemented interesting and useful functionalities:

1. **Flexible multi-camera training:** Choose any combination of cameras for training - single, multiple, or all. You can set these up by **SIMPLY** configuring your selection in the config file.

2. **Powered by gsplat** Integrated [gsplat](https://github.com/nerfstudio-project/gsplat) rasterization kernel with its advanced functions, e.g. absolute gradients, anti-aliasing, etc.

3. **Camera Pose Refinement:** Recognizing that camera poses may not always be sufficiently accurate, we provide a method to refine and optimize these poses.

4. **Objects' GT Bounding Box Refinement:** To address noise in ground truth boxes, we've added this feature to further improve accuracy and robustness.

5. **Affine Transformation:** This feature handles camera exposure and other related issues, enhancing the quality of scene reconstruction. 

6. ...

These functionalities are designed to enhance the overall performance and flexibility of our system, allowing for more accurate and adaptable scene reconstruction across various datasets and conditions.
</details>

## 📢 Updates

**[Jan 2025]**  OmniRe is accepted by ICLR 2025 as a spotlight paper!

**[Aug 2024]**  Release code of DriveStudio!

## 🔨 Installation

Run the following commands to set up the environment:

```shell
# Clone the repository with submodules
git clone --recursive https://github.com/ziyc/drivestudio.git
cd drivestudio

# Create the environment
conda create -n drivestudio python=3.9 -y
conda activate drivestudio
pip install -r requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/NVlabs/nvdiffrast

# Set up for SMPL Gaussians
cd third_party/smplx/
pip install -e .
cd ../..
```

## 🐳 Docker Build
- Build base + NuScenes images with Buildx bake:  
  `docker buildx bake` (uses `docker-bake.hcl`, tags `drivestudio:base` and `drivestudio:nuscenes`)
- Build a single target: `docker buildx bake nuscenes`
- Override tag prefix: `TAG_PREFIX=myrepo/drivestudio docker buildx bake`
- See `docker/README.md` for target details and a sample `docker run` command.

## 📊 Prepare Data
We support most popular public driving datasets. Detailed instructions for downloading and processing each dataset are available in the following documents:

- Waymo: [Data Process Instruction](docs/Waymo.md)
- NuScenes: [Data Process Instruction](docs/NuScenes.md)
- NuPlan: [Data Process Instruction](docs/Nuplan.md)
- ArgoVerse: [Data Process Instruction](docs/ArgoVerse.md)
- PandaSet: [Data Process Instruction](docs/Pandaset.md)
- KITTI: [Data Process Instruction](docs/KITTI.md)

## 🚀 Running
### Training
```shell
export PYTHONPATH=$(pwd)
start_timestep=0 # start frame index for training
end_timestep=-1 # end frame index, -1 for the last frame

python tools/train.py \
    --config_file configs/omnire.yaml \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=waymo/3cams \
    data.scene_idx=$scene_idx \
    data.start_timestep=$start_timestep \
    data.end_timestep=$end_timestep
```

- To run other methods, change `--config_file`. See `configs/` for more options.
- Specify dataset and number of cameras by setting `dataset`. Examples: `waymo/1cams`, `waymo/5cams`, `pandaset/6cams`, `argoverse/7cams`, etc.
  You can set up arbitrary camera combinations for each dataset. See `configs/datasets/` for custom configuration details.
- For over 3 cameras or 450+ images, we recommend using `omnire_extended_cam.yaml`. It works better in practice.
### Evaluation
```shell
python tools/eval.py --resume_from $ckpt_path
```

<details>
<summary>Compare with OmniRe</summary>
To compare with OmniRe, run the following commands to reproduce the paper’s checkpoints.

```
# Scene Reconstruction
for scene_idx in {0..7}; do
    IFS=',' read scene_idx seg_name start_timestep end_timestep <<< $(awk -v idx="$(($scene_idx + 2))" \
        'NR==idx {print $1, $2, $3, $4}' data/waymo_example_scenes.txt)
    
    python tools/train.py \
        --config_file configs/paper_legacy/omnire.yaml \
        --output_root logs/omnire_waymo/ \
        --project recon \
        --run_name ${scene_idx} \
        data.scene_idx=$scene_idx \
        data.start_timestep=$start_timestep \
        data.end_timestep=$end_timestep \
        data.pixel_source.test_image_stride=0
done
```

```
# Novel View Synthesis
for scene_idx in {0..7}; do
    IFS=',' read scene_idx seg_name start_timestep end_timestep <<< $(awk -v idx="$(($scene_idx + 2))" \
        'NR==idx {print $1, $2, $3, $4}' data/waymo_example_scenes.txt)
    
    python tools/train.py \
        --config_file configs/paper_legacy/omnire.yaml \
        --output_root logs/omnire_waymo/ \
        --project nvs \
        --run_name ${scene_idx} \
        data.scene_idx=$scene_idx \
        data.start_timestep=$start_timestep \
        data.end_timestep=$end_timestep \
        data.pixel_source.test_image_stride=10
done
```

and run `utils/gather_results.py` to get the results.
</details>

## 👏 Contributions
We're improving our project to develop a robust driving recom/sim system. Some areas we're focusing on:

- A real-time viewer for background and foreground visualization
- Scene editing and simulation tools
- Other Gaussian representations (e.g., 2DGS, surfels)

We welcome pull requests and collaborations. If you'd like to contribute or have questions, feel free to open an issue or contact [Ziyu Chen](https://github.com/ziyc) (ziyu.sjtu@gmail.com).

## 🙏 Acknowledgments
We utilize the rasterization kernel from [gsplat](https://github.com/nerfstudio-project/gsplat). Parts of our implementation are based on work from [EmerNeRF](https://github.com/NVlabs/EmerNeRF), [NerfStudio](https://github.com/nerfstudio-project/nerfstudio), [GART](https://github.com/JiahuiLei/GART), and [Neuralsim](https://github.com/PJLab-ADG/neuralsim). We've also implemented unofficial versions of [Deformable-GS](https://github.com/ingra14m/Deformable-3D-Gaussians), [PVG](https://github.com/fudan-zvg/PVG), and [Street Gaussians](https://github.com/zju3dv/street_gaussians), with reference to their original codebases.

We extend our deepest gratitude to the authors for their contributions to the community, which have greatly supported our research.

## Citation
If you find this codebase helpful, please kindly cite:
```
@inproceedings{
  chen2025omnire,
  title={OmniRe: Omni Urban Scene Reconstruction},
  author={Ziyu Chen and Jiawei Yang and Jiahui Huang and Riccardo de Lutio and Janick Martinez Esturo and Boris Ivanovic and Or Litany and Zan Gojcic and Sanja Fidler and Marco Pavone and Li Song and Yue Wang},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
