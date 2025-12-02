"""
LCNet (PaddleX PP-LCNet) Pedestrian Attribute Recognition
Renamed from test.py - uses PaddlePaddle's lightweight model
"""
from paddlex import create_model
import os
import time
import re


class LCNetPredictor:
    def __init__(self, model_dir="PP-LCNet_x1_0_pedestrian_attribute_infer"):
        """
        Initialize LCNet predictor
        
        Args:
            model_dir: Path to PaddleX model directory
        """
        print(f"[LCNet] Loading PP-LCNet model from {model_dir}...")
        self.model = create_model(
            model_name="PP-LCNet_x1_0_pedestrian_attribute",
            model_dir=model_dir
        )
        print(f"[LCNet] Model loaded successfully!")
    
    @staticmethod
    def clean_label(label):
        """
        Remove Chinese text inside parentheses:  Something(中文) → Something
        """
        return re.sub(r"\([^)]*\)", "", label).strip()
    
    @staticmethod
    def extract_labels_scores(data):
        """Extract label_names and scores from PaddleX JSON format."""
        if "res" in data:
            block = data["res"]
            if "label_names" in block and "scores" in block:
                return block["label_names"], block["scores"]
        return None, None
    
    def predict(self, image_path, threshold=0.5):
        """
        Predict pedestrian attributes for an image
        
        Args:
            image_path: Path to image file
            threshold: Confidence threshold for predictions
            
        Returns:
            dict with 'labels', 'scores', 'predicted' (above threshold), 'time'
        """
        start_time = time.time()
        
        # Run prediction
        results_iter = self.model.predict(image_path)
        
        # Get first result
        res = next(results_iter, None)
        if res is None:
            return {
                'labels': [],
                'scores': [],
                'predicted': [],
                'time': time.time() - start_time,
                'error': 'No prediction results'
            }
        
        # Extract data
        data = res.json
        label_names, scores = self.extract_labels_scores(data)
        
        if label_names is None:
            return {
                'labels': [],
                'scores': [],
                'predicted': [],
                'time': time.time() - start_time,
                'error': 'Could not extract attributes'
            }
        
        # Clean labels (remove Chinese parts)
        clean_names = [self.clean_label(lbl) for lbl in label_names]
        
        elapsed = time.time() - start_time
        
        # Format results
        results = {
            'labels': clean_names,
            'scores': scores,
            'predicted': [(label, score) for label, score in zip(clean_names, scores) if score >= threshold],
            'time': elapsed
        }
        
        return results


def test_predictor():
    """Test function (original test.py functionality)"""
    predictor = LCNetPredictor()
    
    image_dir = "./crop_result"
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    THRESHOLD = 0.5
    total_txt_path = os.path.join(output_dir, "lcnet_result.txt")
    
    total_runtime = 0.0
    image_count = 0
    
    # Start output file
    with open(total_txt_path, "w", encoding="utf-8") as total_f:
        total_f.write("LCNET PEDESTRIAN ATTRIBUTE INFERENCE RESULTS\n")
        total_f.write("=" * 60 + "\n\n")
    
    # Process images
    for filename in os.listdir(image_dir):
        img_path = os.path.join(image_dir, filename)
        
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        
        image_count += 1
        results = predictor.predict(img_path, threshold=THRESHOLD)
        total_runtime += results['time']
        
        # Write to file
        with open(total_txt_path, "a", encoding="utf-8") as total_f:
            total_f.write(f"Image: {filename}\n")
            total_f.write(f"Processing time: {results['time']:.4f} seconds\n\n")
            
            if 'error' in results:
                total_f.write(f"ERROR: {results['error']}\n")
            else:
                total_f.write(f"Predicted attributes (over {THRESHOLD:.2f}):\n")
                for label, score in results['predicted']:
                    total_f.write(f"  {label}: {score:.4f}\n")
                
                total_f.write("\nAll attribute probabilities:\n")
                sorted_attrs = sorted(zip(results['labels'], results['scores']), 
                                    key=lambda x: x[1], reverse=True)
                for label, score in sorted_attrs:
                    total_f.write(f"  {label}: {score:.4f}\n")
            
            total_f.write("\n" + "-" * 60 + "\n\n")
    
    # Summary
    avg_time = total_runtime / image_count if image_count else 0
    
    with open(total_txt_path, "a", encoding="utf-8") as total_f:
        total_f.write("\nSUMMARY\n")
        total_f.write("=" * 60 + "\n")
        total_f.write(f"Total images processed: {image_count}\n")
        total_f.write(f"Total runtime: {total_runtime:.4f} seconds\n")
        total_f.write(f"Average per image: {avg_time:.4f} seconds\n")
    
    print(f"\nAll results saved to: {total_txt_path}")


if __name__ == "__main__":
    test_predictor()

