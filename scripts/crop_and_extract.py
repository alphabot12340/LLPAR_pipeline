#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run predict_and_crop.py then test.py using given VIS/IR images."
    )
    parser.add_argument(
        "--vis-source",
        required=True,
        help="Path to visible (RGB) image, e.g. ./vis_val/190001.jpg",
    )
    parser.add_argument(
        "--ir-source",
        required=True,
        help="Path to infrared image, e.g. ./Ir_val/190001.jpg",
    )

    args = parser.parse_args()

    # Resolve paths relative to this script's directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    pt_file = os.path.join(base_dir, "weights/deyolo", "best.pt")
    output_dir = os.path.join(base_dir, "crop_result")
    predict_script = os.path.join(base_dir, "predict_and_crop.py")
    test_script = os.path.join(base_dir, "test.py")

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ---- Step 1: run predict_and_crop.py ----
    print("[1/2] Running predict_and_crop.py ...")
    predict_cmd = [
        sys.executable,
        predict_script,
        "--pt-file",
        pt_file,
        "--vis-source",
        args.vis_source,
        "--ir-source",
        args.ir_source,
        "--output",
        output_dir,
        "--conf",
        "0.25",
    ]

    try:
        subprocess.run(predict_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: predict_and_crop.py failed with exit code {e.returncode}")
        sys.exit(e.returncode)

    # ---- Step 2: run test.py ----
    print("[2/2] Running test.py ...")
    test_cmd = [sys.executable, test_script]

    try:
        subprocess.run(test_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: test.py failed with exit code {e.returncode}")
        sys.exit(e.returncode)

    print("Done: cropping + attribute extraction completed.")


if __name__ == "__main__":
    main()
