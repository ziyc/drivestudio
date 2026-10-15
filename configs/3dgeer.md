# Exact and Efficient Fisheye Rendering (3DGEER)

<p align="center">
  <img src="https://github.com/user-attachments/assets/8957f2c7-9888-4cd5-b226-b7aee0cb3ecb" width="80%" style="max-width: 80%; height: auto;" />
</p>

To use 3DGEER with any method:

1. Install the `gsplat-geer` package.

    ```bash
    git clone --branch gsplat-geer https://github.com/boschresearch/3dgeer.git ./third_party/3dgeer
    cd ./third_party/3dgeer

    pip uninstall gsplat
    pip install -e . --no-build-isolation
    cd ../..
    ```

2. In the method's config `.yaml` file, under `trainer.render`, add:

    ```yaml
    render_mode: geer # default | geer
    ```

    Remove `CamPose` from `model`.

    To train on native distorted images, set the following in the dataset config:

    ```yaml
    data:
      pixel_source:
        undistort: False # set True to rectify images to pinhole geometry when loading
    ```

    Native training uses the dataset's camera calibration and requires 3DGEER with support for that camera model.

    PhysicalAI uses one midpoint camera pose per image, a global-shutter approximation.

    To render a fisheye novel-view video, configure `render.render_novel` in the method config:

    ```yaml
    render_novel:
      camera_model: fisheye
      render_mode: geer
      traj_types:
        - front_center_interp
      fov: 180.0 # degrees
      radial_coeffs: [0.0, 0.0, 0.0, 0.0]
      fps: 24
    ```

    To save a separate FTheta novel-view video, add the following under `render`:

    ```yaml
    render_ftheta:
      render_mode: geer
      traj_types:
        - front_center_interp
      camera_id: 0 # dataset camera supplying FTheta calibration
      fps: 24
    ```

    This writes `novel_ftheta_<step>/<trajectory>.mp4` under `videos/` after training or `videos_eval/` during evaluation. Optional `frames`, `height`, and `width` control the video length and resolution. Use `ftheta_parameters` for a calibration JSON path or mapping instead of the source camera's calibration; omit `fov` and `radial_coeffs`. Set `render_ftheta: null` or `False` to disable it. Unspecified trajectory, frame count, and FPS settings inherit from `render_novel` when available.

    An example OmniRe + 3DGEER config can be found [here](./omnire_geer.yaml).

3. Train and evaluate the model according to the [README](../README.md#training).

Note: 3DGEER training requires `gsplat-geer` to return `info["geer_gradient"]` for densification. Its gradient scale differs from default 3DGS, so their densification thresholds are not interchangeable.
