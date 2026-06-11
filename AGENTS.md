# DriveStudio Agent Notes

## Environment
- Prefer `uv` over `README.md`'s older conda/pip setup. The executable source of truth is `pyproject.toml` + `uv.lock`, which require Python `3.10.x`.
- Base install: `uv sync`
- Dataset-prep extras (NuScenes devkit + HF segmentation): `uv sync --group data`
- Quick environment sanity check: `bash scripts/check_env.sh`

## Real Entry Points
- Train one scene: `python tools/train.py --config_file <config> --output_root <dir> --project <name> --run_name <name> dataset=<dataset/cams> data.scene_idx=<idx> [more OmegaConf overrides]`
- Evaluate/render from a checkpoint: `python tools/eval.py --resume_from <log_dir/checkpoint_*.pth> [OmegaConf overrides]`
- Preprocess datasets: `python datasets/preprocess.py ...`
- Extract sky/fine dynamic masks: `python datasets/tools/extract_masks.py ...`

## Checkpoint And Eval Gotchas
- `tools/eval.py` derives `log_dir` from `dirname(--resume_from)` and then loads `config.yaml` from that directory. Do not move checkpoints away from their run directory unless you also move the config.
- Training checkpoints are written directly in the run directory as `checkpoint_<step>.pth` or `checkpoint_final.pth`; there is no `checkpoints/` subdirectory.
- `tools/train.py --resume_from ...` is model-only resume (`load_only_model=True`), not a full optimizer/state resume.
- Resumed training runs do **not** save new checkpoints because checkpoint saving is gated on `args.resume_from is None` in `tools/train.py`.

## Verification
- There is no repo CI/test suite checked in. For code-only changes, use focused verification such as `python -m py_compile <touched files>`.
- For runtime changes, prefer the narrowest real command that exercises the path you touched, e.g. one-scene `tools/train.py`, `tools/eval.py`, or a single-scene mask/preprocess command.

## Repo Structure
- `tools/`: top-level training/eval/export/visualization entry points.
- `configs/`: trainer configs and dataset presets. `dataset=<name/cams>` in CLI resolves to `configs/datasets/<name/cams>.yaml` inside `tools/train.py`.
- `datasets/`: preprocessing plus runtime data loaders. For pose/camera behavior, inspect sourceloaders before changing higher-level code.
- `models/`: trainers and Gaussian/object model implementations.
- `third_party/`: editable vendored deps; `third_party/smplx` is installed from a local path in `pyproject.toml`.

## NuScenes-Specific Traps
- The checked-in NuScenes dataset presets still default to `data/nuscenes/processed_10Hz/mini`. If the workspace uses `data/nuscenes-omni/...`, override `data.data_root=...` on the CLI.
- NuScenes presets enable `load_smpl: True` by default. SMPL runs require `smpl_models/SMPL_NEUTRAL.pkl` (see `docs/NuScenes.md`).
- `datasets/tools/extract_masks.py` now only uses the Hugging Face SegFormer backend; do not reintroduce the old `mmseg`/`segformer_env` workflow.

## Rendering / Viewer
- Novel trajectory rendering is driven from `tools/eval.py` via `render.render_novel.*` config overrides. Built-in trajectory names live in `utils/camera.py`.
- `--enable_viewer` uses `viser`/`nerfview`, but `models/trainers/base.py` documents it as a simple background-only viewer; do not treat it as a full scene inspection tool.
