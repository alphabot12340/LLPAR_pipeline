"""
VTB (VTFPAR++) Pedestrian Attribute Recognition
Uses Vision Transformer with temporal modeling for video-based PAR
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import time

# Add VTFPAR++ to path - this is the project root that contains CLIP/ and models/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VTFPAR_PATH = os.path.join(SCRIPT_DIR, "VTFPAR++")
sys.path.insert(0, VTFPAR_PATH)

# Import CLIP and models exactly as VTFPAR++ does
from CLIP.clip import clip
from models.vit import vit_base

device = "cuda" if torch.cuda.is_available() else "cpu"

# PA100K attributes (matching your training setup)
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


class TransformerClassifier(torch.nn.Module):
    """Simplified VTB model for single image inference"""
    def __init__(self, attr_num, attr_words, dim=768, pretrain_path=None):
        super().__init__()
        self.attr_num = attr_num
        self.word_embed = torch.nn.Linear(512, dim)
        self.visual_embed = torch.nn.Linear(512, dim)
        
        self.vit = vit_base()
        if pretrain_path and os.path.exists(pretrain_path):
            self.vit.load_param(pretrain_path)
        
        self.blocks = self.vit.blocks[-1:]
        self.norm = self.vit.norm
        self.weight_layer = torch.nn.ModuleList([torch.nn.Linear(dim, 1) for _ in range(self.attr_num)])
        self.bn = torch.nn.BatchNorm1d(self.attr_num)
        self.text = clip.tokenize(attr_words).to(device)

    def forward(self, imgs, ViT_model):
        """Forward pass for single or batched images"""
        if len(imgs.size()) == 3:  # Single image (C, H, W)
            imgs = imgs.unsqueeze(0).unsqueeze(0)  # Add batch and frame dims
        elif len(imgs.size()) == 4:  # Batch of images (B, C, H, W)
            imgs = imgs.unsqueeze(1)  # Add frame dimension (B, 1, C, H, W)
        
        batch_size, num_frames, channels, height, width = imgs.size()
        imgs_flat = imgs.view(-1, channels, height, width)
        
        # Extract features using CLIP
        ViT_features = []
        for img in imgs_flat:
            img = img.unsqueeze(0)
            ViT_features.append(ViT_model.encode_image(img).squeeze(0))
        
        ViT_image_features = torch.stack(ViT_features).to(device).float()
        _, token_num, visual_dim = ViT_image_features.size()
        ViT_image_features = ViT_image_features.view(batch_size, num_frames, token_num, visual_dim)
        ViT_image_features = self.visual_embed(torch.mean(ViT_image_features, dim=1))
        
        # Text features
        text_features = ViT_model.encode_text(self.text).to(device).float()
        textual_features = self.word_embed(text_features).expand(
            ViT_image_features.shape[0], text_features.shape[0], 768
        )
        
        # Fusion
        x = torch.cat([textual_features, ViT_image_features], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = torch.cat([self.weight_layer[i](x[:, i, :]) for i in range(self.attr_num)], dim=1)
        
        return logits


class VTBPredictor:
    def __init__(self, checkpoint_path, vit_pretrain_path=None, clip_download_root=None):
        """
        Initialize VTB predictor
        
        Args:
            checkpoint_path: Path to VTB_TEACHER.pth or trained checkpoint
            vit_pretrain_path: Path to jx_vit_base_p16_224-80ecf9dd.pth (optional)
            clip_download_root: Root directory for CLIP model downloads (optional)
        """
        self.device = device
        print(f"[VTB] Loading CLIP model...")
        if clip_download_root:
            self.ViT_model, _ = clip.load("ViT-B/16", device=device, download_root=clip_download_root)
        else:
            self.ViT_model, _ = clip.load("ViT-B/16", device=device)
        self.ViT_model.eval()
        
        print(f"[VTB] Loading VTB model from {checkpoint_path}...")
        self.model = TransformerClassifier(
            len(attr_words), 
            attr_words=attr_words,
            pretrain_path=vit_pretrain_path
        )
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)
        
        self.model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.attr_words = attr_words
        print(f"[VTB] Model loaded successfully!")
    
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
        img_tensor = self.transform(img).to(self.device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(img_tensor, self.ViT_model)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
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
    checkpoint = "./weights/VTB/VTB_TEACHER.pth"
    vit_pretrain = "./weights/jx_vit_base_p16_224-80ecf9dd.pth"
    
    predictor = VTBPredictor(checkpoint, vit_pretrain)
    
    # Test on sample image
    test_image = "./crop_result/test.jpg"
    if os.path.exists(test_image):
        results = predictor.predict(test_image)
        print(f"\nPrediction time: {results['time']:.4f}s")
        print(f"Predicted attributes: {results['predicted']}")


if __name__ == "__main__":
    test_predictor()


