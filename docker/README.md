# DriveStudio Docker

## Build with bake
- Build all: `docker buildx bake` (uses `docker-bake.hcl`)
- Build only NuScenes: `docker buildx bake nuscenes`
- Set a different tag prefix: `TAG_PREFIX=myrepo/drivestudio docker buildx bake`

Targets:
- `base`: Common deps (PyTorch stack, gsplat, pytorch3d, nvdiffrast, smplx, etc.)
- `nuscenes`: Adds `nuscenes-devkit` and a separate SegFormer preprocessing env on top of `base`

## Run NuScenes image
- GPU + data mount: `docker run --gpus all -it --rm -v $(pwd)/data:/workspace/drivestudio/data drivestudio:nuscenes`
- `PYTHONPATH` is preset; work under `/workspace/drivestudio`.
- Prepare NuScenes data per `docs/NuScenes.md` (mount `data/nuscenes/raw`, etc.).
- SegFormer for mask extraction is preinstalled in an isolated conda env (`segformer`) with the code at `/opt/segformer`. Use `conda run -n segformer python datasets/tools/extract_masks.py ...` (or `conda activate segformer`) when running the mask scripts.
