#!/usr/bin/env python3
"""
Pipeline Modes:
    --s : Single image prediction
    --m : Multi image prediction (directory-based pairing)

Output layout:

_PRED_OUTPUT/
    YYYY-MM-DD-HH-MM/
        total_result.txt
        crop_results/
        input_images/          (only if --save-input-images)
            VI/
            IR/
        bounding_boxes/
        overlays/              (only if --save-overlays)
"""

import argparse
import os
import subprocess
import sys
import shutil
from datetime import datetime


def run_cmd(cmd):
    """Run a subprocess command with error handling."""
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed: {cmd}")
        print(f"Exit code: {e.returncode}")
        sys.exit(e.returncode)


def single_image_mode(
    args,
    base_dir,
    predict_script,
    test_script,
    pt_file,
    run_dir,
    crop_dir,
    bbox_dir,
    overlay_dir,
    input_vi_dir,
    input_ir_dir,
):
    """Run detection + crop + attribute recognition for one VIS/IR pair."""
    print("=" * 60)
    print("[MODE] Single Image Mode")
    print("=" * 60)

    # Conditionally save input images
    if args.save_input_images:
        vis_dst = os.path.join(input_vi_dir, os.path.basename(args.vis_source))
        ir_dst = os.path.join(input_ir_dir, os.path.basename(args.ir_source))
        shutil.copy2(args.vis_source, vis_dst)
        shutil.copy2(args.ir_source, ir_dst)

    print("=" * 60)
    print("[STEP 1/2] Detection → Cropping")
    print("=" * 60)

    predict_cmd = [
        sys.executable,
        predict_script,
        "--pt-file", pt_file,
        "--vis-source", args.vis_source,
        "--ir-source", args.ir_source,
        "--output", crop_dir,
        "--conf", str(args.conf),
        "--bbox-dir", bbox_dir,
    ]

    if args.save_overlays:
        predict_cmd += ["--overlay-dir", overlay_dir]

    run_cmd(predict_cmd)

    print("\n" + "=" * 60)
    print("[STEP 2/2] Attribute Recognition")
    print("=" * 60)

    test_cmd = [
        sys.executable,
        test_script,
        "--image-dir", crop_dir,
        "--output-dir", run_dir,
        "--threshold", str(args.threshold),
    ]
    run_cmd(test_cmd)

    print("\nPIPELINE COMPLETE!")
    print(f"Run directory: {run_dir}")
    print(f"  total_result.txt  → {os.path.join(run_dir, 'total_result.txt')}")
    print(f"  crop_results/     → {crop_dir}")
    print(f"  bounding_boxes/   → {bbox_dir}")

    if args.save_input_images:
        print(f"  input_images/VI   → {input_vi_dir}")
        print(f"  input_images/IR   → {input_ir_dir}")

    if args.save_overlays:
        print(f"  overlays/         → {overlay_dir}")

    print("=" * 60)


def multi_image_mode(
    args,
    base_dir,
    predict_script,
    test_script,
    pt_file,
    run_dir,
    crop_dir,
    bbox_dir,
    overlay_dir,
    input_vi_dir,
    input_ir_dir,
):
    """Batch-process VIS/IR image pairs from _PRED_INPUT/VI and IR."""
    print("=" * 60)
    print("[MODE] Multi Image Mode")
    print("=" * 60)

    vi_dir = os.path.join(base_dir, "_PRED_INPUT", "VI")
    ir_dir = os.path.join(base_dir, "_PRED_INPUT", "IR")

    if not os.path.isdir(vi_dir) or not os.path.isdir(ir_dir):
        print(f"ERROR: Expected directories:\n  {vi_dir}\n  {ir_dir}")
        sys.exit(1)

    exts = (".jpg", ".jpeg", ".png", ".bmp")

    vis_files = sorted([f for f in os.listdir(vi_dir) if f.lower().endswith(exts)])
    ir_files = sorted([f for f in os.listdir(ir_dir) if f.lower().endswith(exts)])

    matched = [f for f in vis_files if f in ir_files]

    if not matched:
        print("ERROR: No matching VIS/IR file pairs found in _PRED_INPUT.")
        sys.exit(1)

    print(f"Found {len(matched)} matched VIS/IR pairs\n")

    for idx, filename in enumerate(matched, 1):
        print(f"[{idx}/{len(matched)}] Processing pair: {filename}")

        vis_path = os.path.join(vi_dir, filename)
        ir_path = os.path.join(ir_dir, filename)

        # Conditionally save input images
        if args.save_input_images:
            shutil.copy2(vis_path, os.path.join(input_vi_dir, filename))
            shutil.copy2(ir_path, os.path.join(input_ir_dir, filename))

        predict_cmd = [
            sys.executable,
            predict_script,
            "--pt-file", pt_file,
            "--vis-source", vis_path,
            "--ir-source", ir_path,
            "--output", crop_dir,
            "--conf", str(args.conf),
            "--bbox-dir", bbox_dir,
        ]

        if args.save_overlays:
            predict_cmd += ["--overlay-dir", overlay_dir]

        run_cmd(predict_cmd)

    print("\n" + "=" * 60)
    print("[STEP 2] Attribute Recognition on All Crops")
    print("=" * 60)

    test_cmd = [
        sys.executable,
        test_script,
        "--image-dir", crop_dir,
        "--output-dir", run_dir,
        "--threshold", str(args.threshold),
    ]
    run_cmd(test_cmd)

    print("\nPIPELINE COMPLETE!")
    print(f"Run directory: {run_dir}")
    print(f"  total_result.txt  → {os.path.join(run_dir, 'total_result.txt')}")
    print(f"  crop_results/     → {crop_dir}")
    print(f"  bounding_boxes/   → {bbox_dir}")

    if args.save_input_images:
        print(f"  input_images/VI   → {input_vi_dir}")
        print(f"  input_images/IR   → {input_ir_dir}")

    if args.save_overlays:
        print(f"  overlays/         → {overlay_dir}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run PAR Pipeline")
    parser.add_argument("--s", action="store_true", help="Single image mode")
    parser.add_argument("--m", action="store_true", help="Multi image mode")

    parser.add_argument("--vis-source", help="Path to VIS image (single mode)")
    parser.add_argument("--ir-source", help="Path to IR image (single mode)")

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.25)

    parser.add_argument(
        "--save-input-images",
        action="store_true",
        help="Save VIS/IR input images (default off)",
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Save overlay images (default off)",
    )

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    predict_script = os.path.join(base_dir, "predict_and_crop.py")
    test_script = os.path.join(base_dir, "test.py")
    pt_file = os.path.join(base_dir, "weights", "deyolo", "best.pt")

    # Time stamped run directory
    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    pred_root = os.path.join(base_dir, "_PRED_OUTPUT")
    run_dir = os.path.join(pred_root, ts)

    crop_dir = os.path.join(run_dir, "crop_results")
    bbox_dir = os.path.join(run_dir, "bounding_boxes")
    overlay_dir = os.path.join(run_dir, "overlays")
    input_vi_dir = os.path.join(run_dir, "input_images", "VI")
    input_ir_dir = os.path.join(run_dir, "input_images", "IR")

    # Only create dirs needed for this run
    dirs = [pred_root, run_dir, crop_dir, bbox_dir]

    if args.save_overlays:
        dirs.append(overlay_dir)

    if args.save_input_images:
        dirs.extend([input_vi_dir, input_ir_dir])

    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # Route mode
    if args.s:
        if not args.vis_source or not args.ir_source:
            print("ERROR: --s mode requires --vis-source and --ir-source")
            sys.exit(1)

        single_image_mode(
            args,
            base_dir,
            predict_script,
            test_script,
            pt_file,
            run_dir,
            crop_dir,
            bbox_dir,
            overlay_dir,
            input_vi_dir,
            input_ir_dir,
        )

    elif args.m:
        multi_image_mode(
            args,
            base_dir,
            predict_script,
            test_script,
            pt_file,
            run_dir,
            crop_dir,
            bbox_dir,
            overlay_dir,
            input_vi_dir,
            input_ir_dir,
        )

    else:
        print("ERROR: You must specify either --s or --m")
        sys.exit(1)


if __name__ == "__main__":
    main()
