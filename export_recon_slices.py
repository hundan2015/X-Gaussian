import argparse
import csv
import json
import pickle
import re
import time
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from skimage.metrics import structural_similarity

try:
    from tigre.utilities.geometry import Geometry
except Exception:
    # Delay hard failure to run_asd_pocs() where we report installation issues clearly.
    class Geometry:  # type: ignore
        pass


def window_to_uint8(image, lo, hi):
    image = image.astype(np.float32)
    if hi <= lo:
        return np.zeros_like(image, dtype=np.uint8)
    out = (image - lo) / (hi - lo)
    return np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)


def parse_source_path_from_cfg(cfg_path):
    text = Path(cfg_path).read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"source_path='([^']+)'", text)
    if match:
        return match.group(1)
    return None


def find_latest_ours_render_dir(model_path, split):
    split_dir = Path(model_path) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    ours_dirs = []
    for d in split_dir.iterdir():
        if d.is_dir() and d.name.startswith("ours_"):
            try:
                it = int(d.name.split("_")[-1])
            except ValueError:
                it = -1
            ours_dirs.append((it, d))

    if not ours_dirs:
        raise FileNotFoundError(f"No ours_xxx directory found under {split_dir}")

    ours_dirs.sort(key=lambda x: x[0])
    render_dir = ours_dirs[-1][1] / "renders"
    if not render_dir.exists():
        raise FileNotFoundError(f"Missing renders directory: {render_dir}")
    return render_dir


def load_render_stack(render_dir):
    render_dir = Path(render_dir)
    images = sorted([p for p in render_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not images:
        raise FileNotFoundError(f"No rendered images found in {render_dir}")

    stack = []
    for p in images:
        arr = iio.imread(p)
        arr = arr.astype(np.float32)
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
        # torchvision save_image usually writes [0,255] uint8 png.
        if arr.max() > 1.5:
            arr = arr / 255.0
        stack.append(arr)

    return np.stack(stack, axis=0).astype(np.float32)


class ConeGeometry(Geometry):
    """
    TIGRE-compatible cone-beam geometry (unit converted from mm to m),
    aligned with SAX-NeRF style fields.
    """

    def __init__(self, data):
        super().__init__()
        self.DSD = data["DSD"] / 1000.0
        self.DSO = data["DSO"] / 1000.0

        self.nDetector = np.array(data["nDetector"])
        self.dDetector = np.array(data["dDetector"]) / 1000.0
        self.sDetector = self.nDetector * self.dDetector

        self.nVoxel = np.array(data["nVoxel"])
        self.dVoxel = np.array(data["dVoxel"]) / 1000.0
        self.sVoxel = self.nVoxel * self.dVoxel

        self.offOrigin = np.array(data["offOrigin"]) / 1000.0
        self.offDetector = np.array(data["offDetector"]) / 1000.0

        self.accuracy = data["accuracy"]
        self.mode = data["mode"]
        self.filter = data["filter"]


class DummyDataset:
    """Minimal dataset holder for reconstruction/evaluation flow."""

    def __init__(self, data):
        self.geo = ConeGeometry(data)
        self.train_projs = np.asarray(data["train"]["projections"], dtype=np.float32)
        self.train_angles = np.asarray(data["train"]["angles"], dtype=np.float32)
        self.val_projs = np.asarray(data["val"]["projections"], dtype=np.float32)
        self.val_angles = np.asarray(data["val"]["angles"], dtype=np.float32)
        self.gt_image = np.asarray(data["image"], dtype=np.float32) if "image" in data else None


def run_asd_pocs(projs, geo, angles, niter, lmbda, lmbda_red, init, verbose):
    try:
        import tigre.algorithms as algs
    except Exception as e:
        raise ImportError(
            "Failed to import TIGRE. Install TIGRE first (matching CUDA/Python), then rerun."
        ) from e

    t_start = time.time()
    image = algs.asd_pocs(
        projs,
        geo,
        angles,
        niter=niter,
        lmbda=lmbda,
        lmbda_red=lmbda_red,
        init=init,
        verbose=verbose,
    )
    t_elapsed = time.time() - t_start
    return image, t_elapsed


def orient_like_sax_nerf(volume):
    # Match SAX-NeRF eval_traditional.py orientation convention.
    return np.flip(volume.transpose(2, 1, 0), axis=2)


def maybe_resample_to_256(volume):
    """
    Lightweight trilinear-like separable interpolation to 256^3 without scipy.
    """
    target = 256
    if volume.shape == (target, target, target):
        return volume

    def resize_axis(arr, axis, new_len):
        old_len = arr.shape[axis]
        if old_len == new_len:
            return arr
        old_coords = np.linspace(0.0, 1.0, old_len, dtype=np.float32)
        new_coords = np.linspace(0.0, 1.0, new_len, dtype=np.float32)

        arr_swap = np.moveaxis(arr, axis, 0)
        flat = arr_swap.reshape(old_len, -1)
        out = np.empty((new_len, flat.shape[1]), dtype=np.float32)
        for j in range(flat.shape[1]):
            out[:, j] = np.interp(new_coords, old_coords, flat[:, j]).astype(np.float32)
        out = out.reshape((new_len,) + arr_swap.shape[1:])
        return np.moveaxis(out, 0, axis)

    out = volume.astype(np.float32)
    out = resize_axis(out, 0, target)
    out = resize_axis(out, 1, target)
    out = resize_axis(out, 2, target)
    return out


def robust_normalize(volume, low_q=0.01, high_q=0.995):
    v = volume.astype(np.float32)
    lo = float(np.quantile(v, low_q))
    hi = float(np.quantile(v, high_q))
    if hi <= lo:
        return np.zeros_like(v, dtype=np.float32)
    v = (v - lo) / (hi - lo)
    return np.clip(v, 0.0, 1.0)


def get_psnr_3d(pred, gt, eps=1e-12):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    mse = float(np.mean((pred - gt) ** 2))
    if mse <= eps:
        return float("inf")
    return float(20.0 * np.log10(1.0 / np.sqrt(mse + eps)))


def get_ssim_3d(arr1, arr2, size_average=True, pixel_max=1.0):
    """
    SAX-NeRF style 3D SSIM: compute SSIM on D/H/W axis views and average.
    Expected input shape is [D, H, W] in [0, 1].
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    arr1 = arr1[np.newaxis, ...]
    arr2 = arr2[np.newaxis, ...]

    if not ((arr1.ndim == 4) and (arr2.ndim == 4)):
        raise ValueError(f"Expected [N, D, H, W], got {arr1.shape} and {arr2.shape}")

    arr1 = arr1.astype(np.float64)
    arr2 = arr2.astype(np.float64)
    n = arr1.shape[0]

    arr1_d = np.transpose(arr1, (0, 2, 3, 1))
    arr2_d = np.transpose(arr2, (0, 2, 3, 1))
    ssim_d = np.asarray(
        [
            structural_similarity(arr1_d[i], arr2_d[i], data_range=pixel_max)
            for i in range(n)
        ],
        dtype=np.float64,
    )

    arr1_h = np.transpose(arr1, (0, 1, 3, 2))
    arr2_h = np.transpose(arr2, (0, 1, 3, 2))
    ssim_h = np.asarray(
        [
            structural_similarity(arr1_h[i], arr2_h[i], data_range=pixel_max)
            for i in range(n)
        ],
        dtype=np.float64,
    )

    ssim_w = np.asarray(
        [
            structural_similarity(arr1[i], arr2[i], data_range=pixel_max)
            for i in range(n)
        ],
        dtype=np.float64,
    )

    ssim_avg = (ssim_d + ssim_h + ssim_w) / 3.0
    if size_average:
        return float(ssim_avg.mean())
    return ssim_avg


def load_npy_volume(path, name):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing {name} file: {p}")
    arr = np.load(p)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume for {name}, got shape {arr.shape} from {p}")
    return arr.astype(np.float32)


def update_metrics_3d_csv(model_path, psnr_3d, ssim_3d, asd_pocs_time):
    """
    If metrics_3d.csv exists, create metrics_3d_new.csv with the header and last row,
    replacing legacy psnr/ssim columns with newly computed 3D values, adding explicit
    psnr_3d/ssim_3d columns, and updating pred_time as original pred_time + asd_pocs_time.
    """
    metrics_csv = Path(model_path) / "metrics_3d.csv"
    if not metrics_csv.exists():
        return None

    try:
        with open(metrics_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return None

        # Get header and last row
        fieldnames = list(reader.fieldnames or [])
        last_row = rows[-1]

        # Update psnr column with new psnr_3d value
        if "psnr" in fieldnames and psnr_3d is not None:
            last_row["psnr"] = str(psnr_3d)

        # Update ssim column with new ssim_3d value
        if "ssim" in fieldnames and ssim_3d is not None:
            last_row["ssim"] = str(ssim_3d)

        # # Also write explicit 3D metric columns to avoid ambiguity with 2D metrics.
        # if "psnr_3d" not in fieldnames:
        #     fieldnames.append("psnr_3d")
        # if "ssim_3d" not in fieldnames:
        #     fieldnames.append("ssim_3d")
        # if psnr_3d is not None:
        #     last_row["psnr_3d"] = str(psnr_3d)
        # if ssim_3d is not None:
        #     last_row["ssim_3d"] = str(ssim_3d)

        # Update pred_time: original pred_time + asd_pocs_time
        if "pred_time" in fieldnames and asd_pocs_time is not None:
            try:
                original_pred_time = float(last_row["pred_time"])
                new_pred_time = original_pred_time + asd_pocs_time
                last_row["pred_time"] = str(new_pred_time)
                print(f"[INFO] Updated pred_time: {original_pred_time:.4f}s + {asd_pocs_time:.4f}s = {new_pred_time:.4f}s")
            except (ValueError, TypeError) as e:
                print(f"[WARN] Failed to update pred_time: {e}")

        # Write to metrics_3d_new.csv
        metrics_csv_new = Path(model_path) / "metrics_3d_new.csv"
        with open(metrics_csv_new, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(last_row)

        print(f"[INFO] Created metrics_3d_new.csv with updated psnr_3d/ssim_3d and pred_time values")
        return metrics_csv_new

    except Exception as e:
        print(f"[WARN] Failed to process metrics_3d.csv: {e}")
        return None


def export_slices(volume_norm, output_dir, low_q, high_q, prefix):
    # Keep folder naming close to SAX-NeRF traditional output.
    root_h = Path(output_dir) / "CT" / "H" / prefix
    root_w = Path(output_dir) / "CT" / "W" / prefix
    root_l = Path(output_dir) / "CT" / "L" / prefix
    root_h.mkdir(parents=True, exist_ok=True)
    root_w.mkdir(parents=True, exist_ok=True)
    root_l.mkdir(parents=True, exist_ok=True)

    lo = float(np.quantile(volume_norm, low_q))
    hi = float(np.quantile(volume_norm, high_q))

    h, w, l = volume_norm.shape
    idx_h = np.linspace(0, h - 1, 256).round().astype(np.int32)
    idx_w = np.linspace(0, w - 1, 256).round().astype(np.int32)
    idx_l = np.linspace(0, l - 1, 256).round().astype(np.int32)

    for i, k in enumerate(idx_h):
        iio.imwrite(root_h / f"{prefix}_{i:03d}.png", window_to_uint8(volume_norm[k, ...], lo, hi))
    for i, k in enumerate(idx_w):
        iio.imwrite(root_w / f"{prefix}_{i:03d}.png", window_to_uint8(volume_norm[:, k, :], lo, hi))
    for i, k in enumerate(idx_l):
        iio.imwrite(root_l / f"{prefix}_{i:03d}.png", window_to_uint8(volume_norm[:, :, k], lo, hi))


def resolve_source_pickle(model_path, source_pickle_arg):
    source_pickle = source_pickle_arg
    if source_pickle is None:
        cfg_args = Path(model_path) / "cfg_args"
        if not cfg_args.exists():
            raise FileNotFoundError(
                "Cannot resolve source pickle: missing cfg_args and no --source_pickle provided."
            )
        source_pickle = parse_source_path_from_cfg(cfg_args)

    if source_pickle is None:
        raise ValueError("Failed to parse source_path from cfg_args, please pass --source_pickle explicitly.")

    source_pickle = Path(source_pickle)
    if not source_pickle.exists():
        raise FileNotFoundError(f"Source pickle not found: {source_pickle}")
    return source_pickle


def reconstruct_pred_gt(args, model_path, out_dir):
    source_pickle = resolve_source_pickle(model_path, args.source_pickle)

    with open(source_pickle, "rb") as f:
        data = pickle.load(f)
    dset = DummyDataset(data)

    # Align with SAX-NeRF: flip voxel geometry order before reconstruction.
    dset.geo.nVoxel = np.flip(dset.geo.nVoxel)
    dset.geo.sVoxel = np.flip(dset.geo.sVoxel)
    dset.geo.dVoxel = np.flip(dset.geo.dVoxel)

    if args.projection_source == "pickle_train":
        base_projs = dset.train_projs
        base_angles = dset.train_angles
    elif args.projection_source == "pickle_val":
        base_projs = dset.val_projs
        base_angles = dset.val_angles
    elif args.projection_source == "model_test_renders":
        base_projs = load_render_stack(find_latest_ours_render_dir(model_path, "test"))
        base_angles = dset.val_angles
    else:
        base_projs = load_render_stack(find_latest_ours_render_dir(model_path, "train"))
        base_angles = dset.train_angles

    view_num = min(int(args.nview), base_projs.shape[0], base_angles.shape[0])
    projs = base_projs[:view_num]
    angles = base_angles[:view_num]

    print(f"[INFO] model_path: {model_path}")
    print(f"[INFO] source_pickle: {source_pickle}")
    print(f"[INFO] projection_source: {args.projection_source}")
    print(f"[INFO] using {view_num} projections for ASD-POCS")

    image_pred, asd_pocs_time = run_asd_pocs(
        projs=projs,
        geo=dset.geo,
        angles=angles,
        niter=int(args.niter),
        lmbda=float(args.lmbda),
        lmbda_red=float(args.lmbda_red),
        init=args.init,
        verbose=bool(args.verbose),
    )
    print(f"[INFO] ASD-POCS computation time: {asd_pocs_time:.4f}s")

    image_pred = orient_like_sax_nerf(np.asarray(image_pred, dtype=np.float32))
    image_pred = maybe_resample_to_256(image_pred)
    pred_norm = robust_normalize(image_pred, args.low_q, args.high_q)
    np.save(out_dir / "image_pred.npy", pred_norm.astype(np.float32))
    export_slices(pred_norm, out_dir, args.low_q, args.high_q, prefix="ct_pred")

    gt_norm = None
    if dset.gt_image is not None:
        image_gt = maybe_resample_to_256(np.asarray(dset.gt_image, dtype=np.float32))
        gt_norm = robust_normalize(image_gt, args.low_q, args.high_q)
        np.save(out_dir / "image_gt.npy", gt_norm.astype(np.float32))
        export_slices(gt_norm, out_dir, args.low_q, args.high_q, prefix="ct_gt")

    return {
        "pred_eval": pred_norm,
        "gt_eval": gt_norm,
        "source_pickle": source_pickle,
        "view_num": view_num,
        "asd_pocs_time": asd_pocs_time,
    }


def load_cached_pred_gt(args, out_dir, model_path):
    pred_path = Path(args.pred_npy) if args.pred_npy else out_dir / "image_pred.npy"
    gt_path = Path(args.gt_npy) if args.gt_npy else out_dir / "image_gt.npy"

    print(f"[INFO] model_path: {model_path}")
    print("[INFO] mode: eval_only (skip reconstruction)")
    print(f"[INFO] pred_npy: {pred_path}")
    print(f"[INFO] gt_npy: {gt_path}")

    pred_eval = load_npy_volume(pred_path, "pred")
    gt_eval = load_npy_volume(gt_path, "gt")
    if pred_eval.shape != gt_eval.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred_eval.shape} vs gt {gt_eval.shape}. "
            "Please provide aligned cached volumes."
        )

    return {
        "pred_eval": pred_eval,
        "gt_eval": gt_eval,
        "source_pickle": None,
        "view_num": 0,
        "asd_pocs_time": None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct 3D volume via TIGRE ASD-POCS from an X-Gaussian run directory and export 256 slices."
    )
    parser.add_argument("--model_path", required=True, type=str, help="X-Gaussian output run directory")
    parser.add_argument(
        "--source_pickle",
        default=None,
        type=str,
        help="Optional source pickle path; if omitted, parse from model_path/cfg_args",
    )
    parser.add_argument("--output_dir", default=None, type=str, help="Output directory")
    parser.add_argument("--nview", default=100, type=int, help="Number of train projections used")
    parser.add_argument(
        "--projection_source",
        default="model_test_renders",
        choices=[
            "pickle_train",
            "pickle_val",
            "model_test_renders",
            "model_train_renders",
        ],
        help="Input projection source for ASD-POCS",
    )
    parser.add_argument("--niter", default=6, type=int, help="ASD-POCS iterations")
    parser.add_argument("--lmbda", default=1.0, type=float, help="ASD-POCS lambda")
    parser.add_argument("--lmbda_red", default=0.999, type=float, help="ASD-POCS lambda reduction")
    parser.add_argument("--init", default=None, type=str, help="ASD-POCS init mode")
    parser.add_argument("--verbose", action="store_true", help="Verbose TIGRE solver output")
    parser.add_argument("--low_q", default=0.01, type=float, help="Low quantile for normalization")
    parser.add_argument("--high_q", default=0.995, type=float, help="High quantile for normalization")
    parser.add_argument(
        "--skip_recon",
        action="store_true",
        help="Skip reconstruction and evaluate PSNR/SSIM directly from existing image_pred/image_gt npy files",
    )
    parser.add_argument(
        "--pred_npy",
        default=None,
        type=str,
        help="Optional path to predicted 3D npy file (defaults to output_dir/image_pred.npy)",
    )
    parser.add_argument(
        "--gt_npy",
        default=None,
        type=str,
        help="Optional path to GT 3D npy file (defaults to output_dir/image_gt.npy)",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    out_dir = Path(args.output_dir) if args.output_dir else model_path / "recon_tigre_asd_pocs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_recon:
        stage = load_cached_pred_gt(args, out_dir, model_path)
    else:
        stage = reconstruct_pred_gt(args, model_path, out_dir)

    pred_eval = stage["pred_eval"]
    gt_eval = stage["gt_eval"]
    source_pickle = stage["source_pickle"]
    view_num = stage["view_num"]
    asd_pocs_time = stage["asd_pocs_time"]

    psnr_3d = None
    ssim_3d = None
    if gt_eval is not None:
        psnr_3d = get_psnr_3d(pred_eval, gt_eval)
        ssim_3d = get_ssim_3d(pred_eval, gt_eval)

    metrics = {
        "algorithm": "tigre.asd_pocs",
        "mode": "eval_only" if args.skip_recon else "recon_and_eval",
        "nview": int(view_num),
        "niter": int(args.niter),
        "lmbda": float(args.lmbda),
        "lmbda_red": float(args.lmbda_red),
        "psnr_3d": psnr_3d,
        "ssim_3d": ssim_3d,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Update metrics_3d.csv if it exists
    if psnr_3d is not None:
        update_metrics_3d_csv(model_path, psnr_3d, ssim_3d, asd_pocs_time)

    with open(out_dir / "meta.txt", "w", encoding="utf-8") as f:
        f.write(f"model_path: {model_path}\n")
        f.write(f"mode: {'eval_only' if args.skip_recon else 'recon_and_eval'}\n")
        f.write(f"source_pickle: {source_pickle}\n")
        f.write("algorithm: tigre.asd_pocs\n")
        f.write(f"nview: {view_num}\n")
        f.write(f"projection_source: {args.projection_source}\n")
        f.write(f"niter: {args.niter}\n")
        f.write(f"lmbda: {args.lmbda}\n")
        f.write(f"lmbda_red: {args.lmbda_red}\n")
        f.write(f"init: {args.init}\n")
        f.write(f"psnr_3d: {psnr_3d}\n")
        f.write(f"ssim_3d: {ssim_3d}\n")

    if args.skip_recon:
        print(f"[DONE] Evaluated cached volumes from: {out_dir}")
    else:
        print(f"[DONE] Saved TIGRE ASD-POCS reconstruction to: {out_dir}")
    if psnr_3d is not None:
        print(f"[METRIC] PSNR_3D: {psnr_3d:.6f}")
        print(f"[METRIC] SSIM_3D: {ssim_3d:.6f}")


if __name__ == "__main__":
    main()
