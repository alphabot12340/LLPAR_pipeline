#!/usr/bin/env python3
"""
Pipeline Modes:
    --s : Single image prediction
    --m : Multi image prediction (directory-based pairing)

Model-selection flags:
    --LCNet --VTB --CNN   (optional)
        → If none supplied: ALL THREE MODELS RUN
        → If one or two supplied: only those models run

Ensemble:
    --ensemble
        → Allowed only if:
            (1) all 3 models are selected explicitly, OR
            (2) no model flags were given (meaning all 3 run)

Output layout:

_PRED_OUTPUT/
    YYYY-MM-DD-HH-MM/
        total_result.txt
        crop_results/
        bounding_boxes/
        input_images/ (only if --save-input-images)
        overlays/     (only if --save-overlays)
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


# -----------------------------------------------------------
# SINGLE IMAGE MODE
# -----------------------------------------------------------

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
    model_flags,
):
    print("=" * 60)
    print("[MODE] Single Image Mode")
    print("=" * 60)

    if args.save_input_images:
        os.makedirs(input_vi_dir, exist_ok=True)
        os.makedirs(input_ir_dir, exist_ok=True)
        shutil.copy2(args.vis_source, os.path.join(input_vi_dir, os.path.basename(args.vis_source)))
        shutil.copy2(args.ir_source, os.path.join(input_ir_dir, os.path.basename(args.ir_source)))

    print("=" * 60)
    print("[STEP 1/2] Detection → Cropping")
    print("=" * 60)

    predict_cmd = [
        sys.executable, predict_script,
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
        sys.executable, test_script,
        "--image-dir", crop_dir,
        "--output-dir", run_dir,
        "--threshold", str(args.threshold),
    ]

    # pass what models to run
    test_cmd += model_flags

    # pass ensemble if enabled
    if args.ensemble:
        test_cmd.append("--ensemble")

    run_cmd(test_cmd)

    print("\nPIPELINE COMPLETE!")
    print(f"Run directory: {run_dir}")
    print("=" * 60)


# -----------------------------------------------------------
# MULTI IMAGE MODE
# -----------------------------------------------------------

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
    model_flags,
):
    print("=" * 60)
    print("[MODE] Multi Image Mode")
    print("=" * 60)

    vi_dir = os.path.join(base_dir, "_PRED_INPUT", "VI")
    ir_dir = os.path.join(base_dir, "_PRED_INPUT", "IR")

    if not os.path.isdir(vi_dir) or not os.path.isdir(ir_dir):
        print(f"ERROR: Expected input directories:\n  {vi_dir}\n  {ir_dir}")
        sys.exit(1)

    exts = (".jpg", ".jpeg", ".png", ".bmp")

    vis_files = sorted([f for f in os.listdir(vi_dir) if f.lower().endswith(exts)])
    ir_files = sorted([f for f in os.listdir(ir_dir) if f.lower().endswith(exts)])
    matched = [f for f in vis_files if f in ir_files]

    if not matched:
        print("ERROR: No VIS/IR filename matches.")
        sys.exit(1)

    for filename in matched:
        vis_path = os.path.join(vi_dir, filename)
        ir_path = os.path.join(ir_dir, filename)

        if args.save_input_images:
            os.makedirs(input_vi_dir, exist_ok=True)
            os.makedirs(input_ir_dir, exist_ok=True)
            shutil.copy2(vis_path, os.path.join(input_vi_dir, filename))
            shutil.copy2(ir_path, os.path.join(input_ir_dir, filename))

        predict_cmd = [
            sys.executable, predict_script,
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
        sys.executable, test_script,
        "--image-dir", crop_dir,
        "--output-dir", run_dir,
        "--threshold", str(args.threshold),
    ]

    test_cmd += model_flags

    if args.ensemble:
        test_cmd.append("--ensemble")

    run_cmd(test_cmd)

    print("\nPIPELINE COMPLETE!")
    print("=" * 60)


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run PAR Pipeline")

    # pipeline modes
    parser.add_argument("--s", action="store_true", help="Single image mode")
    parser.add_argument("--m", action="store_true", help="Multi image mode")

    # input paths
    parser.add_argument("--vis-source")
    parser.add_argument("--ir-source")

    # runtime settings
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--conf", type=float, default=0.25)

    # optional saving
    parser.add_argument("--save-input-images", action="store_true")
    parser.add_argument("--save-overlays", action="store_true")

    # model-selection flags
    parser.add_argument("--LCNet", action="store_true")
    parser.add_argument("--VTB", action="store_true")
    parser.add_argument("--CNN", action="store_true")

    # ensemble flag
    parser.add_argument("--ensemble", action="store_true")

    args = parser.parse_args()

    # -------------------------------------------------------
    # MODEL SELECTION LOGIC
    # -------------------------------------------------------
    model_flags = []

    selected = {
        "LCNet": args.LCNet,
        "VTB": args.VTB,
        "CNN": args.CNN,
    }

    num_selected = sum(selected.values())

    if num_selected == 0:
        # user selected no model flags → run all
        model_flags = ["--LCNet", "--VTB", "--CNN"]

    else:
        for m, enabled in selected.items():
            if enabled:
                model_flags.append(f"--{m}")

    # Ensemble restriction:
    if args.ensemble:
        if num_selected not in (0, 3):
            print("ERROR: --ensemble requires ALL base models.\n"
                  "Fix by:\n"
                  "  → remove model-selection flags (default: all 3 run)\n"
                  "  OR\n"
                  "  → explicitly set: --LCNet --VTB --CNN")
            sys.exit(1)

    # -------------------------------------------------------
    # PATH SETUP
    # -------------------------------------------------------

    base_dir = os.path.dirname(os.path.abspath(__file__))
    predict_script = os.path.join(base_dir, "predict_and_crop.py")
    test_script = os.path.join(base_dir, "test.py")
    pt_file = os.path.join(base_dir, "weights", "deyolo", "best.pt")

    ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_dir = os.path.join(base_dir, "_PRED_OUTPUT", ts)

    crop_dir = os.path.join(run_dir, "crop_results")
    bbox_dir = os.path.join(run_dir, "bounding_boxes")
    overlay_dir = os.path.join(run_dir, "overlays")
    input_vi_dir = os.path.join(run_dir, "input_images", "VI")
    input_ir_dir = os.path.join(run_dir, "input_images", "IR")

    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(bbox_dir, exist_ok=True)
    if args.save_overlays:
        os.makedirs(overlay_dir, exist_ok=True)

    # -------------------------------------------------------
    # MODE EXECUTION
    # -------------------------------------------------------

    if args.s:
        if not args.vis_source or not args.ir_source:
            print("ERROR: --s requires --vis-source and --ir-source.")
            sys.exit(1)

        single_image_mode(
            args, base_dir, predict_script, test_script, pt_file,
            run_dir, crop_dir, bbox_dir, overlay_dir,
            input_vi_dir, input_ir_dir, model_flags
        )

    elif args.m:
        multi_image_mode(
            args, base_dir, predict_script, test_script, pt_file,
            run_dir, crop_dir, bbox_dir, overlay_dir,
            input_vi_dir, input_ir_dir, model_flags
        )

    else:
        print("ERROR: You must specify either --s or --m")
        sys.exit(1)


if __name__ == "__main__":
    main()
