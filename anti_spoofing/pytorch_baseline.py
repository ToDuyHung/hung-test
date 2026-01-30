import os
import argparse
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN

# Import local MiniFASNet
# Ensure the current directory is in sys.path so we can import src
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model_lib.MiniFASNet import MiniFASNetV2, MiniFASNetV2SE
from src.utility import get_kernel
from src.generate_patches import CropImage
from src.data_io import transform as trans

def load_mini_fasnet(model_path, device, width=80, height=80, num_classes=3):
    # Calculate kernel size for the final layer based on input size
    # For 80x80, this should be (5, 5). 
    # Logic from utility.py: kernel_size = ((height + 15) // 16, (width + 15) // 16)
    kernel_size = get_kernel(height, width)
    
    # Load model
    model = MiniFASNetV2(conv6_kernel=kernel_size, num_classes=num_classes)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        # Adapt keys if necessary (remove 'module.' prefix if present)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        print(f"[WARN] Model weights not found at {model_path}. Initializing random weights.")
        print(f"       Please download weights: 2.7_80x80_MiniFASNetV2.pth")
    
    model.to(device)
    model.eval()
    return model

class AntiSpoofPredictor:
    def __init__(self, model_path, device_id="cuda:0"):
        self.device = torch.device(device_id if torch.cuda.is_available() else "cpu")
        self.input_size = (80, 80)
        self.scale = 2.7
        
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        
        # Initialize Anti-Spoof Model
        self.model = load_mini_fasnet(model_path, self.device)
        
        # Official Cropper
        self.image_cropper = CropImage()
        # Official Transform
        self.test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        
    def preprocess_image(self, img, bbox):
        # Image is BGR (H, W, 3)
        # bbox is [x1, y1, x2, y2]
        # CropImage expects [x, y, w, h]
        
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        bbox_wh = [x1, y1, w, h]
        
        # Use official CropImage logic (checks aspect ratio internally if scale used?)
        # crop(self, org_img, bbox, scale, out_w, out_h, crop=True)
        img_cropped = self.image_cropper.crop(img, bbox_wh, self.scale, 80, 80)
        
        # Transform
        img_tensor = self.test_transform(img_cropped)
        return img_tensor.unsqueeze(0).to(self.device)

    def predict(self, img_path):
        # Load image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Failed to load {img_path}")
            return None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 1. Detect Face
        # MTCNN expects RGB
        boxes, probs = self.mtcnn.detect(img_rgb)
        
        if boxes is None or len(boxes) == 0:
            return []
        
        results = []
        for i, box in enumerate(boxes):
            # 2. Preprocess (Crop scale 2.7 -> 80x80)
            # IMPORTANT: Model was trained on BGR images (standard cv2), 
            # so we must pass img_bgr, even though MTCNN used RGB.
            inp = self.preprocess_image(img_bgr, box)
            
            # 3. Forward
            with torch.no_grad():
                logits = self.model(inp) # (1, 3)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Class 0: Spoof, Class 1: Real (usually). Need to verify index mapping.
            # In Silent-Face-Anti-Spoofing:
            # 2 classes: 0 - Spoof, 1 - Live (Real).
            # But num_classes=3 in MiniFASNet?
            # Actually, the repo default is num_classes=3.
            # label 0: spoof, label 1: live? 
            # Or is it (spoof, live, other)?
            # The repo says:
            # "class 0: spoof, class 1: real" in `data/data_merge.py`?
            # Actually `anti_spoof_predict.py` takes `prediction[:, 1]` as real score.
            # So index 1 is Real. Index 0 is Spoof.
            
            score_real = probs[1]
            label = "Real" if score_real > 0.5 else "Spoof"
            score = score_real if label == "Real" else (1 - score_real)
            
            results.append({
                "box": box, 
                "label": label, 
                "score": score,
                "probs": probs
            })
            
        return results

    def export_coreml(self, output_path="MiniFASNetV2.mlpackage"):
        import coremltools as ct
        
        print(f"Exporting model to {output_path}...")
        
        # Prepare dummy input
        dummy_input = torch.rand(1, 3, 80, 80).to(self.device)
        
        # Trace
        traced_model = torch.jit.trace(self.model, dummy_input)
        
        # Convert
        # Using ImageType for easy iOS integration
        # Scale: 1/255 if the model expects [0,1].
        # If the model expected [0, 255], scale=1.0.
        # My preprocessing used / 255.0. So the CoreML input should be normalized.
        # coremltools ImageType implicitly is 0-255? No, it passes normalized or not.
        # color_layout: RGB
        
        # We want the mlmodel to take a CVPixelBuffer (which is 0-255) and normalize it to 0-1.
        # So scale=1/255.
        
        classifier_config = ct.ClassifierConfig(class_labels=["Spoof", "Real", "Other"])
        
        model = ct.convert(
            traced_model,
            inputs=[ct.ImageType(name="input_1", shape=dummy_input.shape, scale=1/255.0, color_layout="RGB")],
            # classifier_config=classifier_config, # Optional, if we want strict classification
        )
        
        model.save(output_path)
        print("Export success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None, help="Path to image or folder")
    parser.add_argument("--model_path", type=str, default="2.7_80x80_MiniFASNetV2.pth")
    parser.add_argument("--export_coreml", action="store_true", help="Export to CoreML")
    args = parser.parse_args()
    
    # Check if model exists, if not, try to download or warn
    if not os.path.exists(args.model_path):
        # Try to download from a known mirror if possible or just warn
        # For now, we assume user or I downloaded it via curl
        pass

    predictor = AntiSpoofPredictor(args.model_path)
    
    if args.export_coreml:
        predictor.export_coreml("MiniFASNetV2.mlpackage")
        sys.exit(0)
        
    if args.data_root:
        if os.path.isdir(args.data_root):
            files = [os.path.join(args.data_root, f) for f in os.listdir(args.data_root) 
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        else:
            files = [args.data_root]
            
        for f in files:
            res = predictor.predict(f)
            print(f"File: {os.path.basename(f)}")
            for r in res:
                print(f"  Label: {r['label']} (Score: {r['score']:.4f}) | Box: {r['box']}")
    else:
        print("Please provide --data_root to run inference, or --export_coreml")
