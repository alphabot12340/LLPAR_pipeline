"""
CNN (ResNet18 Student) Pedestrian Attribute Recognition
Uses ResNet18 student model trained with knowledge distillation
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import time

# Add SNN-PAR to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SNN_PAR_PATH = os.path.join(SCRIPT_DIR, "SNN-PAR")
sys.path.insert(0, SNN_PAR_PATH)

from models_CNNHELPERS.base_block import TransformerClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

# PA100K attributes
attr_words = [
    'Female', 'AgeOver60', 'Age18-60', 'AgeLess18',
    'Front', 'Side', 'Back',
    'Hat', 'Glasses',
    'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront',
    'ShortSleeve', 'LongSleeve',
    'UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice',
    'LowerStripe', 'LowerPattern', 'LongCoat',
    'Trousers', 'Shorts', 'Skirt&Dress',
    'boots'
]


class CNNPredictor:
    def __init__(self, checkpoint_path, vit_pretrain_path=None):
        """
        Initialize CNN (ResNet18 Student) predictor
        
        Args:
            checkpoint_path: Path to CNN_STUDENT.pth
            vit_pretrain_path: Path to jx_vit_base_p16_224-80ecf9dd.pth (optional, for teacher)
        """
        self.device = device
        print(f"[CNN] Loading ResNet18 Student model from {checkpoint_path}...")
        
        # Create model
        if vit_pretrain_path and os.path.exists(vit_pretrain_path):
            self.model = TransformerClassifier(
                len(attr_words),
                pretrain_path=vit_pretrain_path,
                use_resnet18=True
            )
        else:
            print("[CNN] Warning: ViT pretrain not found, loading without teacher weights")
            self.model = TransformerClassifier(
                len(attr_words),
                pretrain_path='',
                use_resnet18=True
            )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'state_dicts' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dicts'], strict=False)
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.model.eval()
        
        # Get label vectors for model input: dummy zeros of shape (attr_num, 768)
        self.label_vector = np.zeros((len(attr_words), 768), dtype=np.float32)
        self.label_v_tensor = torch.from_numpy(self.label_vector).to(device)
        self.label_v_tensor = torch.from_numpy(self.label_vector).to(device)
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.attr_words = attr_words
        print(f"[CNN] Model loaded successfully!")
    
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
        
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            # Model returns: T_train_logits, T_img_features, text_features, S_train_logits, S_img_features
            outputs = self.model(img_tensor, self.label_v_tensor)
            S_train_logits = outputs[3]  # Student logits
            probs = torch.sigmoid(S_train_logits).squeeze().cpu().numpy()
        
        elapsed = time.time() - start_time
        
        # Format results
        results = {
            'labels': self.attr_words,
            'scores': probs.tolist() if isinstance(probs, np.ndarray) else [probs],
            'predicted': [(label, score) for label, score in zip(self.attr_words, probs) if score >= threshold],
            'time': elapsed
        }
        
        return results


def test_predictor():
    """Test function"""
    checkpoint = "./weights/resnet-18/CNN_STUDENT.pth"
    vit_pretrain = "./weights/jx_vit_base_p16_224-80ecf9dd.pth"
    
    predictor = CNNPredictor(checkpoint, vit_pretrain)
    
    # Test on sample image
    test_image = "./crop_result/test.jpg"
    if os.path.exists(test_image):
        results = predictor.predict(test_image)
        print(f"\nPrediction time: {results['time']:.4f}s")
        print(f"Predicted attributes: {results['predicted']}")


if __name__ == "__main__":
    test_predictor()

