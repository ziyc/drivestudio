"""UrbanV2X preprocessing: ROS1 bag (+ GNSS/INS ground truth, calibration) -> drivestudio scene folder.

Raw layout (see docs/UrbanV2X.md)
  <data_root>/bags/<bag>.bag                       ego vehicle sensors (cameras, LiDAR, IMU)
  <data_root>/calibration/{intrinsics,extrinsics}/  per-sensor calibration package (Kalibr convention)
  <data_root>/gt/<gt>.txt                          Inertial Explorer GNSS/INS export (UTCTime = Unix UTC, 100 Hz)
Scene list (one scene per line, '#' = comment):  <scene_name> <bag> <gt_file>
Integer scene ids (datasets/preprocess.py --scene_ids) are line indices into this list; names work too.

Output scene folder (same layout as the KITTI loader)
  images/{t:03d}_{cam}.jpg    undistorted, rescaled (default 0.5 -> 1100x870)
  intrinsics/{cam}.txt        fx fy cx cy 0 0 0 0 0 (distortion already removed)
  extrinsics/{cam}.txt        cam -> ego, OpenCV camera axes; ego = /imu/data frame
  ego_pose/{t:03d}.txt        ego -> world (local ENU anchored at the first GT epoch of the clip)
  lidar/{t:03d}.bin           float32 x y z intensity, ego frame at the camera time, per-point deskewed
  meta.json                   frames, timing, anchor, refined body->IMU extrinsic and residuals
A timestep is one frame of the reference camera (cam 2); other cameras are matched by nearest header time.
The GNSS/INS body -> IMU extrinsic X and the sensor clock offset dt are refined by multi-scan LiDAR consistency
(point-to-plane between deskewed scans >= 0.5 s apart; both sides depend on X, so the fit does not anchor to its start).
"""
import glob, json, logging, os
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

logger = logging.getLogger()

CAMERAS = list(range(7))
REF_CAM = 2
LIDAR_TOPIC, LIDAR_SENSOR = "/velodyne_points", "hdl32"
CAM_TOPIC = "/hikcamera/image_{}/compressed"
_PC2 = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


def pointcloud2(msg):
    d = np.dtype({"names": [f.name for f in msg.fields], "formats": [_PC2[f.datatype] for f in msg.fields],
                  "offsets": [f.offset for f in msg.fields], "itemsize": msg.point_step})
    return np.frombuffer(bytes(msg.data), dtype=d)


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def lla2ecef(lat, lon, h):
    A, E2 = 6378137.0, 6.69437999014e-3
    la, lo = np.radians(lat), np.radians(lon)
    N = A / np.sqrt(1 - E2 * np.sin(la) ** 2)
    return np.stack([(N + h) * np.cos(la) * np.cos(lo), (N + h) * np.cos(la) * np.sin(lo), (N * (1 - E2) + h) * np.sin(la)], -1)


def ecef2enu_rot(lat0, lon0):
    la, lo = np.radians(lat0), np.radians(lon0)
    return np.array([[-np.sin(lo), np.cos(lo), 0],
                     [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                     [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])


def voxel_idx(P, v):
    _, i = np.unique(np.floor(P / v).astype(np.int64), axis=0, return_index=True)
    return i


class UrbanV2XProcessor:
    def __init__(self, load_dir, save_dir, scene_file, process_keys=("images", "lidar", "calib", "pose"),
                 process_id_list=None, workers=1, scale=0.5, refine_extrinsic=True):
        self.load_dir, self.save_dir, self.process_keys = load_dir, save_dir, list(process_keys)
        self.scale, self.refine_extrinsic = scale, refine_extrinsic
        rows = [l.split() for l in open(scene_file) if l.strip() and not l.startswith("#")]
        self.scenes = {r[0]: dict(bag=r[1], gt=r[2]) for r in rows}
        self.names = [r[0] for r in rows]       # integer scene ids (datasets/preprocess.py) index this list
        self.process_id_list = [self._name(i) for i in process_id_list] if process_id_list else list(self.names)
        if workers != 1:
            logger.info("UrbanV2X preprocessing runs scenes sequentially (one bag pass per scene); ignoring workers")

    def _name(self, scene):
        return self.names[int(scene)] if str(scene).isdigit() else str(scene)

    def __len__(self):
        return len(self.process_id_list)

    def convert(self):
        for scene in self.process_id_list:
            self.convert_one(scene)

    # ------------------------------------------------------------------ one scene
    def convert_one(self, scene):
        scene = self._name(scene); s = self.scenes[scene]; out = os.path.join(self.save_dir, scene); cache = os.path.join(out, "_cache")
        for d in ("images", "intrinsics", "extrinsics", "ego_pose", "lidar", "_cache/lidar"):
            os.makedirs(os.path.join(out, d), exist_ok=True)
        calib = os.path.join(self.load_dir, "calibration")
        logger.info(f"[{scene}] extracting {s['bag']}")
        frames = self.extract(os.path.join(self.load_dir, "bags", s["bag"]), calib, out, cache)
        if any(k in self.process_keys for k in ("pose", "lidar", "calib")):
            self.poses(os.path.join(self.load_dir, "gt", s["gt"]), calib, out, cache, frames)

    # ------------------------------------------------------------------ bag -> images, cached scans
    def extract(self, bag, calib, out, cache):
        from rosbags.highlevel import AnyReader          # pip install rosbags
        maps = {}
        for c in CAMERAS:
            y = yaml.safe_load(open(f"{calib}/intrinsics/intrinsics_cam{c}.yaml"))
            K = np.array(y["cameraMatrix"]["data"]).reshape(3, 3); D = np.array(y["distCoeffs"]["data"])
            W, H = y["imageWidth"], y["imageHeight"]
            newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (W, H), 0)          # alpha=0: no black border
            w, h = int(round(W * self.scale)), int(round(H * self.scale))
            Ks = newK.copy(); Ks[0] *= w / W; Ks[1] *= h / H
            maps[c] = cv2.initUndistortRectifyMap(K, D, None, Ks, (w, h), cv2.CV_32FC1)
            np.savetxt(f"{out}/intrinsics/{c}.txt", [Ks[0, 0], Ks[1, 1], Ks[0, 2], Ks[1, 2], 0, 0, 0, 0, 0])
        cam_t = {c: [] for c in CAMERAS}; velo = []
        topics = {CAM_TOPIC.format(c): c for c in CAMERAS}
        want_img = "images" in self.process_keys
        with AnyReader([Path(bag)]) as r:
            cons = [x for x in r.connections if x.topic in topics or x.topic == LIDAR_TOPIC]
            for con, t, raw in r.messages(connections=cons):
                m = r.deserialize(raw, con.msgtype)
                if con.topic in topics:
                    c = topics[con.topic]; k = len(cam_t[c]); cam_t[c].append(stamp(m))
                    if want_img:
                        im = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)
                        im = cv2.remap(im, *maps[c], cv2.INTER_AREA if self.scale < 1 else cv2.INTER_LINEAR)
                        cv2.imwrite(f"{cache}/_cam{c}_{k:03d}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])
                else:
                    x = pointcloud2(m); ok = np.isfinite(x["x"])
                    np.savez(f"{cache}/lidar/{len(velo):03d}.npz", xyz=np.stack([x["x"], x["y"], x["z"]], 1)[ok],
                             intensity=x["intensity"][ok], t_abs=stamp(m) + x["time"][ok].astype(np.float64))
                    velo.append(stamp(m))
        ref = np.array(cam_t[REF_CAM]); frames = []
        for ti, tr in enumerate(ref):
            row = {"t": ti, "cam_hdr": {}}
            for c in CAMERAS:
                ct = np.array(cam_t[c]); k = int(np.argmin(abs(ct - tr))); row["cam_hdr"][str(c)] = float(ct[k])
                src = f"{cache}/_cam{c}_{k:03d}.jpg"
                if want_img and os.path.exists(src):
                    os.replace(src, f"{out}/images/{ti:03d}_{c}.jpg")
            frames.append(row)
        for f in glob.glob(f"{cache}/_cam*.jpg"):
            os.remove(f)
        spread = [max(f["cam_hdr"].values()) - min(f["cam_hdr"].values()) for f in frames]
        logger.info(f"timesteps {len(ref)}, lidar scans {len(velo)}, "
                    f"camera spread median {1e3 * np.median(spread):.1f} ms")
        json.dump(dict(frames=frames, n_lidar=len(velo)), open(f"{cache}/frames.json", "w"))
        return frames

    # ------------------------------------------------------------------ GNSS/INS -> ego poses, deskewed lidar
    def poses(self, gt_file, calib, out, cache, frames):
        TREF = frames[0]["cam_hdr"][str(REF_CAM)]        # times relative to TREF: Unix seconds lose d/d(dt) in float64
        T0, T1 = TREF - 3, frames[-1]["cam_hdr"][str(REF_CAM)] + 3
        rows = []
        for l in open(gt_file):
            s = l.split()
            if len(s) > 24 and s[0][0].isdigit():
                t = float(s[0])
                if T0 <= t <= T1: rows.append([t, float(s[3]), float(s[4]), float(s[5]), float(s[21]), float(s[22]), float(s[23])])
        G = np.array(rows); G[:, 0] -= TREF
        anc = dict(latitude_deg=float(G[0, 1]), longitude_deg=float(G[0, 2]), altitude_m=float(G[0, 3]))
        Re = ecef2enu_rot(anc["latitude_deg"], anc["longitude_deg"])
        ENU = (lla2ecef(G[:, 1], G[:, 2], G[:, 3]) - lla2ecef(anc["latitude_deg"], anc["longitude_deg"], anc["altitude_m"])) @ Re.T
        HEAD, ROLL, PITCH = np.unwrap(np.radians(G[:, 6])), np.radians(G[:, 4]), np.radians(G[:, 5])

        def gt_pose(t):   # IE body: x right, y forward, z up; heading clockwise from north; R = Rz(-psi) Rx(theta) Ry(phi)
            t = np.atleast_1d(t)
            pos = np.stack([np.interp(t, G[:, 0], ENU[:, k]) for k in range(3)], 1)
            psi, th, ph = (np.interp(t, G[:, 0], v) for v in (HEAD, PITCH, ROLL))
            return Rot.from_euler("ZXY", np.stack([-psi, th, ph], 1)).as_matrix(), pos

        def load_T(f):
            y = yaml.safe_load(open(f)); T = np.eye(4)
            T[:3, :3] = np.array(y["Rotation"]).reshape(3, 3); T[:3, 3] = y["Translation"]; return T, y
        T_IL, yl = load_T(f"{calib}/extrinsics/LiDAR-IMU/{LIDAR_SENSOR}_to_imu.yaml")
        DT_L = list(yl["time_offset_to_imu_s_by_dataset"].values())[-1]
        T_CI, DT_C = {}, 0.0
        for c in CAMERAS:
            T_CI[c], yc = load_T(f"{calib}/extrinsics/Camera-IMU/imu_to_cam{c}.yaml")
            DT_C = list(yc["timeshift_cam_imu_by_dataset"].values())[-1]

        scans = []
        for f in sorted(glob.glob(f"{cache}/lidar/*.npz")):
            z = np.load(f); xyz, ta = z["xyz"], z["t_abs"] - TREF + DT_L
            r = np.linalg.norm(xyz, axis=1); ok = (r > 3.0) & (r < 60)
            scans.append(dict(xyz=xyz[ok], t=ta[ok], tmid=float(np.median(ta)), xyz_all=xyz, t_all=ta, inten_all=z["intensity"], r_all=r))

        def X_of(x):
            T = np.eye(4); T[:3, :3] = Rot.from_rotvec(x[:3]).as_matrix(); T[:3, 3] = x[3:6]; return T

        def to_enu(pts, t, x):
            Xm = X_of(x) @ T_IL; pb = pts @ Xm[:3, :3].T + Xm[:3, 3]; R, pos = gt_pose(t + x[6])
            return np.einsum("nij,nj->ni", R, pb) + pos

        rng = np.random.default_rng(0); SEL = list(range(0, len(scans), 4)); MIN_GAP = 5

        def build_pairs(x, n_max=40000):
            P, T, S = [], [], []
            for i in SEL:
                s = scans[i]; k = voxel_idx(to_enu(s["xyz"], s["t"], x), 0.25)
                P.append(s["xyz"][k]); T.append(s["t"][k]); S.append(np.full(len(k), i))
            P, T, S = np.concatenate(P), np.concatenate(T), np.concatenate(S)
            W = to_enu(P, T, x); tree = cKDTree(W); q = rng.choice(len(W), min(len(W), 4 * n_max), replace=False)
            d, nb = tree.query(W[q], k=24, distance_upper_bound=1.0); out_ = []
            for qi, row, dr in zip(q, nb, d):
                row = row[np.isfinite(dr)]; row = row[abs(S[row] - S[qi]) >= MIN_GAP]
                if len(row) < 6: continue
                nbp = W[row]; c = nbp.mean(0); _, sv, vt = np.linalg.svd(nbp - c, full_matrices=False)
                if sv[2] > 0.3 * sv[1]: continue
                out_.append((qi, row[np.argmin(np.linalg.norm(nbp - W[qi], axis=1))], vt[2]))
                if len(out_) >= n_max: break
            qi, j, n = (np.array(v) for v in zip(*out_))
            return dict(Pa=P[qi], Ta=T[qi], Pb=P[j], Tb=T[j], n=n)

        def residual(x, pr, prior=True):
            r = np.einsum("ni,ni->n", pr["n"], to_enu(pr["Pa"], pr["Ta"], x) - to_enu(pr["Pb"], pr["Tb"], x))
            return np.concatenate([r, np.sqrt(len(r)) * 0.02 * np.r_[x[3:6] / 1.0, x[6] / 0.1]]) if prior else r

        x = np.zeros(7); log = []
        if self.refine_extrinsic:
            for it in range(1, 7):
                pr = build_pairs(x)
                x = least_squares(residual, x, args=(pr,), loss="huber", f_scale=0.1,
                                  x_scale=[0.01] * 3 + [0.1] * 3 + [0.01], diff_step=1e-4).x
                r = np.abs(residual(x, pr, prior=False)); log.append(dict(iter=it, median_m=float(np.median(r)), x=x.tolist()))
                logger.info(f"extrinsic refinement {it}: median {np.median(r):.4f} m, dt {1e3 * x[6]:.1f} ms")
        prf = build_pairs(x)
        stats = lambda xx: float(np.median(np.abs(residual(xx, prf, prior=False))))
        X = X_of(x)

        def ego_T(t):
            R, pos = gt_pose(t + x[6]); T = np.eye(4); T[:3, :3] = R[0]; T[:3, 3] = pos[0]; return T @ X

        for c in CAMERAS:
            np.savetxt(f"{out}/extrinsics/{c}.txt", np.linalg.inv(T_CI[c]))
        tmid = np.array([s["tmid"] for s in scans]); ego_times = []
        for f in frames:
            t = f["t"]; te = f["cam_hdr"][str(REF_CAM)] - TREF + DT_C; ego_times.append(te + TREF)
            E = ego_T(te); np.savetxt(f"{out}/ego_pose/{t:03d}.txt", E)
            s = scans[int(np.argmin(abs(tmid - te)))]; keep = (s["r_all"] > 2.0) & (s["r_all"] < 120)
            pe = (to_enu(s["xyz_all"][keep], s["t_all"][keep], x) - E[:3, 3]) @ E[:3, :3]
            np.hstack([pe, s["inten_all"][keep, None] / 255.0]).astype(np.float32).tofile(f"{out}/lidar/{t:03d}.bin")
        meta = dict(enu_anchor=anc, t_ref_unix=TREF, X_body_imu=X.tolist(), dt_gt_minus_sensor_s=float(x[6]),
                    cam_timeshift_s=DT_C, lidar_timeshift_s=DT_L, refine_log=log,
                    final_pairs=dict(identity=dict(median_m=stats(np.zeros(7))), refined=dict(median_m=stats(x))),
                    frames_ego_time_unix=ego_times, ref_camera=REF_CAM, image_scale=self.scale)
        json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
        logger.info(f"multi-scan residual: identity {meta['final_pairs']['identity']['median_m']:.3f} m -> "
                    f"refined {meta['final_pairs']['refined']['median_m']:.3f} m")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True); p.add_argument("--target_dir", required=True)
    p.add_argument("--split_file", default="data/urbanv2x_example_scenes.txt"); p.add_argument("--scene_ids", nargs="*")
    p.add_argument("--process_keys", nargs="+", default=["images", "lidar", "calib", "pose"])
    p.add_argument("--scale", type=float, default=0.5); p.add_argument("--no_refine", action="store_true")
    a = p.parse_args()
    UrbanV2XProcessor(a.data_root, a.target_dir, a.split_file, a.process_keys, a.scene_ids, 1, a.scale, not a.no_refine).convert()
