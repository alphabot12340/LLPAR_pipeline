from ultralytics import YOLO
import argparse
import os
from pathlib import Path
import sys
import cv2
import numpy as np



# CURRENT TEST USAGE:
# cd DEYOLO
# conda activate csc490-deyolo
# python ./scripts/predict_and_crop.py --pt-file "./runs/detect/train16/weights/best.pt" --vis-source "./data/LLVIP_yolo/images/vis_val/190001.jpg" --ir-source "./data/LLVIP_yolo/images/Ir_val/190001.jpg" --conf 0.25







try:
    from tqdm import tqdm
    _HAVE_TQDM = True
except Exception:
    tqdm = None
    _HAVE_TQDM = False

DEFAULT_PT_FILE = "DEYOLO/runs/detect/train16/weights/best.pt"
DEFAULT_CONF_THRESHOLD = 0.25


def clamp(v, a, b):
    return max(a, min(b, v))


def predict_and_crop(pt_file: str, vis_source: str, ir_source: str, out_root: str, conf: float, exts=None, show_progress: bool = True):
    if exts is None:
        exts = ('.jpg', '.jpeg', '.png', '.bmp')

    pt_file = os.path.expanduser(pt_file)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {pt_file}")
    model = YOLO(pt_file)

    # Resolve vis/ir sources and build pair list
    vis_path = Path(os.path.expanduser(vis_source))
    ir_path = Path(os.path.expanduser(ir_source))

    pairs = []  # list of (vis_path, ir_path)

    # Both directories -> pair by filename
    if vis_path.is_dir() and ir_path.is_dir():
        vis_files = sorted([p for p in vis_path.iterdir() if p.suffix.lower() in exts])
        for vf in vis_files:
            counterpart = ir_path / vf.name
            if counterpart.exists():
                pairs.append((vf, counterpart))
            else:
                print(f"Warning: no counterpart for {vf} in {ir_path}, skipping")
    # Both files -> single pair
    elif vis_path.is_file() and ir_path.is_file():
        pairs.append((vis_path, ir_path))
    # One dir, one file -> attempt to find matching filename in dir
    elif vis_path.is_dir() and ir_path.is_file():
        candidate = vis_path / ir_path.name
        if candidate.exists():
            pairs.append((candidate, ir_path))
        else:
            # if no match, try all files in vis dir paired with same-stem matching
            for vf in sorted([p for p in vis_path.iterdir() if p.suffix.lower() in exts]):
                if vf.stem == ir_path.stem:
                    pairs.append((vf, ir_path))
    elif vis_path.is_file() and ir_path.is_dir():
        candidate = ir_path / vis_path.name
        if candidate.exists():
            pairs.append((vis_path, candidate))
        else:
            for rf in sorted([p for p in ir_path.iterdir() if p.suffix.lower() in exts]):
                if rf.stem == vis_path.stem:
                    pairs.append((vis_path, rf))
    else:
        # fallback: treat inputs as globs/streams or single-named strings
        # try to expand glob for vis and ir
        import glob
        vis_matches = sorted(glob.glob(str(vis_source)))
        ir_matches = sorted(glob.glob(str(ir_source)))
        # pair by basename
        for v in vis_matches:
            b = Path(v).name
            match = [i for i in ir_matches if Path(i).name == b]
            if match:
                pairs.append((Path(v), Path(match[0])))

    if not pairs:
        print("No pairs found for given vis/ir sources. Exiting.")
        return 0, 0

    print(f"Running prediction on {len(pairs)} paired inputs, conf={conf}")

    # progress bar setup (per-pair)
    use_tqdm = _HAVE_TQDM and show_progress
    pbar = tqdm(total=len(pairs), unit='pair', desc='Predicting') if use_tqdm else None

    # fallback single-line status writer
    def write_status(msg: str):
        sys.stdout.write('\r\x1b[2K' + msg)
        sys.stdout.flush()

    total_images = 0
    total_crops = 0

    # Derive source_root from visible source if it's a directory (for relative output paths)
    source_root = vis_path if vis_path.is_dir() else None

    for vis_p, ir_p in pairs:
        # status
        if pbar:
            pbar.set_postfix_str(vis_p.name, refresh=False)
        else:
            write_status(f'Processing: {vis_p.name}')

        # call model on the pair
        results = model.predict(source=[str(vis_p), str(ir_p)], conf=conf, save=False, verbose=False)
        if not results:
            if pbar:
                pbar.update(1)
            continue
        res = results[0]

        # prefer to load visible image for cropping to guarantee expected coordinate space
        img = cv2.imread(str(vis_p))
        if img is None:
            if pbar:
                pbar.update(1)
            continue

        total_images += 1

        # extract boxes: try xyxy (x1,y1,x2,y2)
        boxes = []
        try:
            xyxy = getattr(res.boxes, 'xyxy', None)
            if xyxy is not None:
                if hasattr(xyxy, 'cpu'):
                    xyxy = xyxy.cpu().numpy()
                else:
                    xyxy = np.array(xyxy)
                boxes = xyxy
        except Exception:
            boxes = []

        if boxes is None or len(boxes) == 0:
            if pbar:
                pbar.update(1)
            continue

        h, w = img.shape[:2]

        # Determine output directory for this image (preserve relative subfolders if available)
        if source_root is not None:
            try:
                rel = vis_p.relative_to(source_root)
                out_dir = out_root / rel.parent
            except Exception:
                out_dir = out_root
        else:
            out_dir = out_root
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, row in enumerate(boxes):
            x1 = int(clamp(int(row[0]), 0, w - 1))
            y1 = int(clamp(int(row[1]), 0, h - 1))
            x2 = int(clamp(int(row[2]), 0, w - 1))
            y2 = int(clamp(int(row[3]), 0, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]

            out_name = f"{vis_p.stem}_pred{i}{vis_p.suffix}"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), crop)
            total_crops += 1

        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()
    else:
        print()

    print(f"Images scanned: {total_images}, crops created: {total_crops}")
    return total_images, total_crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict with YOLO and crop detections (paired vis+ir inputs)")
    parser.add_argument('--pt-file', '-m', default=DEFAULT_PT_FILE, help=f'Path to model weights (.pt). Default: {DEFAULT_PT_FILE}')
    parser.add_argument('--vis-source', '-v', required=True, help='Visible source (file, dir, glob or stream)')
    parser.add_argument('--ir-source', '-r', required=True, help='Infrared source (file, dir, glob or stream)')
    parser.add_argument('--output', '-o', default=str(Path(__file__).resolve().parents[1] / 'runs' / 'predict_crops'), help='Output root directory for crops')
    parser.add_argument('--conf', '-c', type=float, default=DEFAULT_CONF_THRESHOLD, help=f'Detection confidence threshold. Default: {DEFAULT_CONF_THRESHOLD}')
    parser.add_argument('--progress', dest='progress', action='store_true', help='Show tqdm progress bar (if available)')
    parser.add_argument('--no-progress', dest='progress', action='store_false', help="Don't show progress bar")
    parser.set_defaults(progress=True)
    args = parser.parse_args()

    predict_and_crop(args.pt_file, args.vis_source, args.ir_source, args.output, args.conf, show_progress=args.progress)
