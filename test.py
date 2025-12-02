"""
test.py — Attribute inference stage of PAR pipeline

Supports:
    --LCNet
    --VTB
    --CNN
    --ensemble
"""

import os
import time
import argparse

from Ensemble_Models_Predict import vote_ensemble, moe_ensemble


# ----------------------------------------------------------
# Optional model imports (loaded only if requested)
# ----------------------------------------------------------

def load_lcnet():
    from LCNetPredict import LCNetPredictor
    return LCNetPredictor("PP-LCNet_x1_0_pedestrian_attribute_infer")

def load_vtb():
    from VTBPredict import VTBPredictor
    return VTBPredictor(
        checkpoint_path="./weights/VTB/VTB_TEACHER.pth",
        vit_pretrain_path="./weights/jx_vit_base_p16_224-80ecf9dd.pth",
    )

def load_cnn():
    from CNNPredict import CNNPredictor
    return CNNPredictor(
        checkpoint_path="./weights/resnet-18/CNN_STUDENT.pth",
        vit_pretrain_path="./weights/jx_vit_base_p16_224-80ecf9dd.pth",
    )


# ----------------------------------------------------------
# Formatting helper
# ----------------------------------------------------------

def format_model_section(name, results, threshold):
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append(f"MODEL: {name}")
    lines.append("=" * 60)

    time_val = results["time"]
    if isinstance(time_val, float):
        lines.append(f"Processing time: {time_val:.4f} seconds\n")
    else:
        lines.append(f"Processing time: {time_val}\n")

    # predictions
    lines.append(f"Predicted attributes (over {threshold:.2f}):")
    if results["predicted"]:
        for label, score in results["predicted"]:
            if isinstance(score, float):
                lines.append(f"  {label}: {score:.4f}")
            else:
                lines.append(f"  {label}: {score}")
    else:
        lines.append("  (none)")

    lines.append("\nAll attribute probabilities:")
    sorted_attrs = sorted(zip(results["labels"], results["scores"]), key=lambda x: x[1], reverse=True)

    for label, score in sorted_attrs:
        if isinstance(score, float):
            lines.append(f"  {label}: {score:.4f}")
        else:
            lines.append(f"  {label}: {score}")

    return lines


# ----------------------------------------------------------
# MAIN PROCESS FUNCTION
# ----------------------------------------------------------

def process_images(image_dir, output_dir, threshold, run_lc, run_vt, run_cn, run_ensemble):

    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, "total_result.txt")

    image_files = sorted([f for f in os.listdir(image_dir)
                          if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])

    if not image_files:
        print("No valid images in:", image_dir)
        return

    # Load models only if requested
    lcnet = load_lcnet() if run_lc else None
    vtbn  = load_vtb()   if run_vt else None
    cnnm  = load_cnn()   if run_cn else None

    # Summary timers
    total_times = {"lc": 0, "vt": 0, "cn": 0}

    with open(outfile, "w", encoding="utf-8") as f:
        f.write("MULTI-MODEL PEDESTRIAN ATTRIBUTE RECOGNITION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write("Models: ")
        if run_lc: f.write("LCNet  ")
        if run_vt: f.write("VTB  ")
        if run_cn: f.write("CNN  ")
        if run_ensemble: f.write("| Ensembles (Vote + MoE)")
        f.write("\n")
        f.write(f"Total images: {len(image_files)}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write("=" * 60 + "\n")

    # MAIN LOOP
    for idx, img_name in enumerate(image_files, 1):
        path = os.path.join(image_dir, img_name)
        print(f"[{idx}/{len(image_files)}] Processing:", img_name)

        # base model outputs
        lc_res = vt_res = cn_res = None
        overall_start = time.time()

        if run_lc:
            lc_res = lcnet.predict(path, threshold)
            total_times["lc"] += lc_res["time"]

        if run_vt:
            vt_res = vtbn.predict(path, threshold)
            total_times["vt"] += vt_res["time"]

        if run_cn:
            cn_res = cnnm.predict(path, threshold)
            total_times["cn"] += cn_res["time"]

        overall_time = time.time() - overall_start

        # ensembles
        if run_ensemble:
            vote_res = vote_ensemble(lc_res, vt_res, cn_res, threshold)
            moe_res  = moe_ensemble(lc_res, vt_res, cn_res)

        with open(outfile, "a", encoding="utf-8") as f:

            f.write("\n\n" + "=" * 60 + "\n")
            f.write(f"IMAGE {idx}/{len(image_files)}: {img_name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total processing time: {overall_time:.4f} sec\n")

            if run_lc:
                for line in format_model_section("LCNet", lc_res, threshold):
                    f.write(line + "\n")

            if run_vt:
                for line in format_model_section("VTB", vt_res, threshold):
                    f.write(line + "\n")

            if run_cn:
                for line in format_model_section("CNN", cn_res, threshold):
                    f.write(line + "\n")

            if run_ensemble:
                for line in format_model_section("Vote Ensemble", vote_res, threshold):
                    f.write(line + "\n")
                for line in format_model_section("MoE Ensemble", moe_res, threshold):
                    f.write(line + "\n")

    print("\nPROCESSING COMPLETE!")
    print("Results saved to:", outfile)


# ----------------------------------------------------------
# ENTRYPOINT
# ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image-dir", default="./crop_results")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--LCNet", action="store_true")
    parser.add_argument("--VTB", action="store_true")
    parser.add_argument("--CNN", action="store_true")

    parser.add_argument("--ensemble", action="store_true")

    args = parser.parse_args()

    # Determine models to run
    selected = [args.LCNet, args.VTB, args.CNN]
    num_selected = sum(selected)

    # If none selected → run all
    if num_selected == 0:
        run_lc = run_vt = run_cn = True

    else:
        run_lc = args.LCNet
        run_vt = args.VTB
        run_cn = args.CNN

    # Ensemble restriction
    if args.ensemble:
        if num_selected not in (0, 3):
            print("ERROR: --ensemble requires ALL base models.\n"
                  "Fix by:\n"
                  "  → remove model flags (run all models), OR\n"
                  "  → set: --LCNet --VTB --CNN")
            sys.exit(1)

    process_images(
        args.image_dir, 
        args.output_dir,
        args.threshold,
        run_lc,
        run_vt,
        run_cn,
        args.ensemble
    )


if __name__ == "__main__":
    main()
