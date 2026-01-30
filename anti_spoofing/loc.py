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
    def __init__(self, model_path, device_id="cuda:0", use_coreml=False, use_onnx=False):
        self.device = torch.device(device_id if torch.cuda.is_available() else "cpu")
        self.input_size = (80, 80)
        self.scale = 2.7
        self.use_coreml = use_coreml
        self.use_onnx = use_onnx
        
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        
        if self.use_coreml:
             self.load_coreml_model(model_path)
             self.model = None
        elif self.use_onnx:
             self.load_onnx_model(model_path)
             self.model = None
        else:
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
        
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        bbox_wh = [x1, y1, w, h]
        
        # crop(self, org_img, bbox, scale, out_w, out_h, crop=True)
        img_cropped = self.image_cropper.crop(img, bbox_wh, self.scale, 80, 80)
        
        if self.use_coreml:
            # STRATEGY 3: Raw Tensor Input.
            # We don't return a PIL Image. We return the preprocessed Numpy Tensor directly.
            # img_cropped is BGR (H, W, C)
            
            # 1. Resize if needed (cropper does resize, so 80x80 already)
            # 2. Normalize: The src/data_io/functional.py DOES NOT DIVIDE BY 255.
            # It returns float tensor in [0, 255].
            # So we keep it as float [0, 255].
            img_float = img_cropped.astype(np.float32)
            
            # 3. Transpose to (C, H, W) -> (3, 80, 80)
            img_chw = img_float.transpose(2, 0, 1)
            
            # 4. Add Batch Dim -> (1, 3, 80, 80)
            img_batch = np.expand_dims(img_chw, axis=0)
            
            return img_batch # Return numpy array

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
            inp = self.preprocess_image(img_bgr, box)
            
            # 3. Forward
            probs_np = None
            # 3. Forward
            probs_np = None
            if self.use_coreml:
                # inp is PIL Image
                logits = self.predict_coreml(inp)
                if logits is not None:
                    # logits might be (1, 3) or just (3,)
                    if len(logits.shape) == 1:
                        logits = np.expand_dims(logits, axis=0)
                    
                    # Apply softmax if model output is raw logits
                    import scipy.special
                    probs_np = scipy.special.softmax(logits, axis=1)[0]
            elif self.use_onnx:
                logits = self.predict_onnx(inp)
                import scipy.special
                probs_np = scipy.special.softmax(logits, axis=1)[0]
            else:
                with torch.no_grad():
                    logits = self.model(inp) # (1, 3)
                    probs_np = F.softmax(logits, dim=1).cpu().numpy()[0]
            
            if probs_np is None:
                continue

            # Class 0: Spoof, Class 1: Real
            score_real = probs_np[1]
            label = "Real" if score_real > 0.5 else "Spoof"
            score = score_real if label == "Real" else (1 - score_real)
            
            results.append({
                "box": box, 
                "label": label, 
                "score": score,
                "probs": probs_np
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
        classifier_config = ct.ClassifierConfig(class_labels=["Spoof", "Real", "Other"])
        
        model = ct.convert(
            traced_model,
            # STRATEGY 3: TensorType
            # removing scale/color_layout. We assume input is already valid float tensor.
            inputs=[ct.TensorType(name="input_1", shape=dummy_input.shape)],
            compute_precision=ct.precision.FLOAT32,
            # classifier_config=classifier_config, # Optional, if we want strict classification
        )
        
        model.save(output_path)
        print("Export success!")

    def load_coreml_model(self, model_path):
        import coremltools as ct
        print(f"Loading Core ML model from {model_path}...")
        self.coreml_model = ct.models.MLModel(model_path)
        self.use_coreml = True

    def predict_coreml(self, img_np):
        # img_np is (1, 3, 80, 80) numpy array (BGR order)
        try:
            prediction = self.coreml_model.predict({'input_1': img_np})
            
            # The output name depends on the model. Usually 'var_xxx' or user text.
            # We need to inspect the model or assume standard output.
            # Torch conversion usually gives 'var_xx'.
            # Let's handle generic dictionary return.
            
            # We expect a dictionary with output tensor/array
            # Let's look for the probability output
            
            # Common output names from torch conversion: "linear_0", "var_46", etc.
            # We will just take the first value that looks like the output tensor
            output_tensor = None
            for k, v in prediction.items():
                if hasattr(v, 'shape') or isinstance(v, (np.ndarray, list)):
                    output_tensor = v
                    break
            
            if output_tensor is None:
                print(f"Could not find output tensor in prediction keys: {prediction.keys()}")
                return None
            
            return output_tensor # resulting numpy array
            
        except Exception as e:
            print(f"Core ML prediction error: {e}")
            return None


    def export_onnx(self, output_path="MiniFASNetV2.onnx"):
        print(f"Exporting model to {output_path}...")
        dummy_input = torch.rand(1, 3, 80, 80).to(self.device)
        
        # Opset 11 or 12 is usually safe for Core ML conversion later
        torch.onnx.export(self.model, 
                          dummy_input, 
                          output_path, 
                          verbose=True, 
                          input_names=['input_1'], 
                          output_names=['output_1'],
                          opset_version=12)
        print("ONNX Export success!")

    def load_onnx_model(self, model_path):
        import onnxruntime
        print(f"Loading ONNX model from {model_path}...")
        self.onnx_session = onnxruntime.InferenceSession(model_path)
        self.use_onnx = True

    def predict_onnx(self, img_tensor):
        # img_tensor is (1, 3, 80, 80) tensor on device.
        # ONNX Runtime expects numpy on CPU.
        inp_np = img_tensor.cpu().numpy()
        
        inputs = {self.onnx_session.get_inputs()[0].name: inp_np}
        logits = self.onnx_session.run(None, inputs)[0]
        return logits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None, help="Path to image or folder")
    parser.add_argument("--model_path", type=str, default="2.7_80x80_MiniFASNetV2.pth")
    parser.add_argument("--export_coreml", action="store_true", help="Export to Core ML")
    parser.add_argument("--export_onnx", action="store_true", help="Export to ONNX")
    parser.add_argument("--use_coreml", action="store_true", help="Use Core ML model for inference")
    parser.add_argument("--use_onnx", action="store_true", help="Use ONNX model for inference")
    args = parser.parse_args()
    
    # Check if model exists, if not, try to download or warn
    if not os.path.exists(args.model_path):
        # Try to download from a known mirror if possible or just warn
        # For now, we assume user or I downloaded it via curl
        pass

    # If using ONNX or CoreML, we might pass a different file extension in model_path
    # But usually we pass the .pth for export, and .onnx/.mlpackage for inference.
    # The AntiSpoofPredictor logic needs to know which one to load.
    
    predictor = AntiSpoofPredictor(args.model_path, use_coreml=args.use_coreml, use_onnx=args.use_onnx)
    
    if args.export_coreml:
        predictor.export_coreml("MiniFASNetV2.mlpackage")
        sys.exit(0)
        
    if args.export_onnx:
        predictor.export_onnx("MiniFASNetV2.onnx")
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
        print("Please provide --data_root to run inference, or --export_coreml, or --export_onnx")
