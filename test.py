"""
Multi-Model Pedestrian Attribute Recognition Pipeline
Orchestrates LCNet, VTB, and CNN predictions on cropped images
"""
import os
import time
from LCNetPredict import LCNetPredictor
from VTBPredict import VTBPredictor
from CNNPredict import CNNPredictor


def format_model_section(model_name, results, threshold=0.5):
    """Format results for a single model"""
    lines = []
    lines.append(f"\n{'=' * 60}")
    lines.append(f"MODEL: {model_name}")
    lines.append(f"{'=' * 60}")
    lines.append(f"Processing time: {results['time']:.4f} seconds\n")
    
    if 'error' in results:
        lines.append(f"ERROR: {results['error']}\n")
        return lines
    
    # Predicted attributes (above threshold)
    lines.append(f"Predicted attributes (over {threshold:.2f}):")
    if results['predicted']:
        for label, score in results['predicted']:
            lines.append(f"  {label}: {score:.4f}")
    else:
        lines.append(f"  (none above threshold)")
    
    # All attributes sorted by score
    lines.append("\nAll attribute probabilities:")
    sorted_attrs = sorted(zip(results['labels'], results['scores']), 
                         key=lambda x: x[1], reverse=True)
    for label, score in sorted_attrs:
        lines.append(f"  {label}: {score:.4f}")
    
    return lines


def process_images(image_dir="./crop_result", output_dir="./output", threshold=0.5):
    """
    Process all images with all three models
    
    Args:
        image_dir: Directory containing cropped images
        output_dir: Directory for output files
        threshold: Confidence threshold for predictions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize models
    print("=" * 60)
    print("Initializing models...")
    print("=" * 60)
    
    lcnet = LCNetPredictor("PP-LCNet_x1_0_pedestrian_attribute_infer")
    vtb = VTBPredictor(
        checkpoint_path="./weights/VTB/VTB_TEACHER.pth",
        vit_pretrain_path="./weights/jx_vit_base_p16_224-80ecf9dd.pth"
    )
    cnn = CNNPredictor(
        checkpoint_path="./weights/resnet-18/CNN_STUDENT.pth",
        vit_pretrain_path="./weights/jx_vit_base_p16_224-80ecf9dd.pth"
    )
    
    print("\n" + "=" * 60)
    print("All models loaded successfully!")
    print("=" * 60 + "\n")
    
    # Output file
    total_txt_path = os.path.join(output_dir, "total_result.txt")
    
    # Get all images
    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]
    image_files.sort()
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images to process\n")
    
    # Initialize output file
    with open(total_txt_path, "w", encoding="utf-8") as f:
        f.write("MULTI-MODEL PEDESTRIAN ATTRIBUTE RECOGNITION RESULTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Models: LCNet (PaddleX), VTB (VTFPAR++), CNN (ResNet18)\n")
        f.write(f"Total images: {len(image_files)}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write("=" * 60 + "\n")
    
    # Process each image
    total_time = {
        'lcnet': 0.0,
        'vtb': 0.0,
        'cnn': 0.0,
        'overall': 0.0
    }
    
    for idx, filename in enumerate(image_files, 1):
        img_path = os.path.join(image_dir, filename)
        
        print(f"[{idx}/{len(image_files)}] Processing: {filename}")
        
        overall_start = time.time()
        
        # Run all models
        lcnet_results = lcnet.predict(img_path, threshold)
        vtb_results = vtb.predict(img_path, threshold)
        cnn_results = cnn.predict(img_path, threshold)
        
        overall_time = time.time() - overall_start
        
        total_time['lcnet'] += lcnet_results['time']
        total_time['vtb'] += vtb_results['time']
        total_time['cnn'] += cnn_results['time']
        total_time['overall'] += overall_time
        
        # Write to file
        with open(total_txt_path, "a", encoding="utf-8") as f:
            f.write("\n\n" + "=" * 60 + "\n")
            f.write(f"IMAGE {idx}/{len(image_files)}: {filename}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Total processing time: {overall_time:.4f} seconds\n")
            
            # LCNet section
            for line in format_model_section("LCNet (PaddleX)", lcnet_results, threshold):
                f.write(line + "\n")
            
            # VTB section
            for line in format_model_section("VTB (VTFPAR++)", vtb_results, threshold):
                f.write(line + "\n")
            
            # CNN section
            for line in format_model_section("CNN (ResNet18 Student)", cnn_results, threshold):
                f.write(line + "\n")
        
        print(f"  ✓ LCNet: {lcnet_results['time']:.3f}s | VTB: {vtb_results['time']:.3f}s | CNN: {cnn_results['time']:.3f}s | Total: {overall_time:.3f}s")
    
    # Write summary
    num_images = len(image_files)
    with open(total_txt_path, "a", encoding="utf-8") as f:
        f.write("\n\n" + "=" * 60 + "\n")
        f.write("FINAL SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total images processed: {num_images}\n")
        f.write(f"\nTotal runtime by model:\n")
        f.write(f"  LCNet:   {total_time['lcnet']:.4f} seconds ({total_time['lcnet']/num_images:.4f}s per image)\n")
        f.write(f"  VTB:     {total_time['vtb']:.4f} seconds ({total_time['vtb']/num_images:.4f}s per image)\n")
        f.write(f"  CNN:     {total_time['cnn']:.4f} seconds ({total_time['cnn']/num_images:.4f}s per image)\n")
        f.write(f"  Overall: {total_time['overall']:.4f} seconds ({total_time['overall']/num_images:.4f}s per image)\n")
        f.write(f"\nAverage time per image: {total_time['overall']/num_images:.4f} seconds\n")
        f.write("=" * 60 + "\n")
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Total images: {num_images}")
    print(f"Total time: {total_time['overall']:.2f}s")
    print(f"Average per image: {total_time['overall']/num_images:.4f}s")
    print(f"\nResults saved to: {total_txt_path}")
    print("=" * 60)


def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(
        description="Run multi-model pedestrian attribute recognition"
    )
    parser.add_argument(
        "--image-dir",
        default="./crop_result",
        help="Directory containing cropped images (default: ./crop_result)"
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory (default: ./output)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    process_images(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        threshold=args.threshold
    )


if __name__ == "__main__":
    main()

