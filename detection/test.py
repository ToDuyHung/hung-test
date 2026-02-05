import os
import argparse
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN

# ------------------------------------------------------------------------------
# 1. Utils
# ------------------------------------------------------------------------------
def load_rgb_uint8(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.uint8)

def resize_by_max_side(img, max_side):
    h, w = img.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return img, 1.0
    scale = max_side / float(max(h, w))
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return img_resized, scale

# ------------------------------------------------------------------------------
# 2. ONNX Wrapper (copied/adapted from onnx_baseline.py)
# ------------------------------------------------------------------------------
class GenericONNXModel(nn.Module):
    def __init__(self, onnx_path, name=""):
        super().__init__()
        import onnxruntime
        self.name = name or os.path.basename(onnx_path)
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = onnxruntime.InferenceSession(
            onnx_path, 
            sess_options=sess_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.dummy_param = nn.Parameter(torch.tensor(0.0)) # For device/dtype checks if needed

    def forward(self, x):
        x_np = x.detach().cpu().numpy().astype(np.float32)
        outs = self.session.run(self.output_names, {self.input_name: x_np})
        outs_torch = [torch.from_numpy(o) for o in outs]
        if len(outs_torch) == 1:
            return outs_torch[0]
        return tuple(outs_torch)

# ------------------------------------------------------------------------------
# 3. CoreML Wrapper
# ------------------------------------------------------------------------------
class GenericCoreMLModel(nn.Module):
    def __init__(self, mlmodel_path, name=""):
        super().__init__()
        import coremltools as ct
        self.name = name or os.path.basename(mlmodel_path)
        print(f"Loading CoreML model: {mlmodel_path}")
        self.model = ct.models.MLModel(mlmodel_path)
        self.dummy_param = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # x: torch.Tensor, typically (Batch, C, H, W)
        x_np = x.detach().cpu().numpy().astype(np.float32)
        
        # CoreML prediction usually expects dictionary input
        # Note: 'input_1' is a common default name, but we should verify names during export
        # For this wrapper, we assume the input name used during export was 'input' or 'input_1'
        # To be robust, we'll try to find the input name from the model spec if possible, 
        # or assume the single input.
        
        # Simplified: using the first input description
        input_desc = self.model.input_description
        input_name = next(iter(input_desc)) if input_desc else "input_1"
        
        # Do prediction
        # CoreML expects straight numpy arrays for TensorType inputs
        try:
            preds = self.model.predict({input_name: x_np})
        except Exception as e:
            # Fallback if batch dimension issue, though typically we handle batches 
            # carefully. MTCNN PNet is 1-batch usually in local pyramid, but RNet/ONet are batched.
            # CoreML batch prediction can be tricky.
            # If batch > 1 and model doesn't support it, we might loop.
            # But let's assume we exported with batch size or flexible shapes.
            raise e

        # preds is a dict {output_name: value}
        # We need to return them in the correct order expected by MTCNN
        # PNet: (probs, offsets) usually, or (offsets, probs)?
        # MTCNN PyTorch PNet return: (offsets, probs) ie (bbox_reg, classifier)
        # RNet return: (offsets, probs)
        # ONet return: (offsets, probs, landmarks)
        
        # We need to map keys back to positional arguments
        # We will assume key naming convention or sort.
        # Let's verify output names logic. 
        # Usually checking output shapes is robust enough for MTCNN since they differ.
        # PNet: [1, 4, H, W], [1, 2, H, W]
        # RNet: [N, 4], [N, 2]
        # ONet: [N, 4], [N, 2], [N, 10]
        
        values = list(preds.values())
        
        # Convert to torch
        values_torch = [torch.from_numpy(v) for v in values]
        
        # We need to strictly order them.
        # Sort by output name is one way if we controlled naming.
        # Or sorting by shape size? 
        # Let's rely on naming if we export them with specific names.
        
        # HACK: For now, sort dictionary by key to ensure deterministic order, 
        # assuming export used "output_1", "output_2" or similar.
        sorted_keys = sorted(preds.keys())
        sorted_values = [preds[k] for k in sorted_keys]
        values_torch = [torch.from_numpy(v) for v in sorted_values]
        
        if len(values_torch) == 1:
            return values_torch[0]
        return tuple(values_torch)

# ------------------------------------------------------------------------------
# 4. Export Functionality
# ------------------------------------------------------------------------------
class Rank4PReLU(nn.Module):
    def __init__(self, original_prelu):
        super().__init__()
        self.prelu = original_prelu
        
    def forward(self, x):
        # x is (N, C) or (N, C, 1, 1) or whatever.
        # But specifically for the failing case, it's (N, C).
        # We force it to (N, C, 1, 1) to satisfy CoreML PReLU constraints for non-scalar alpha.
        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
            x = self.prelu(x)
            x = x.squeeze(-1).squeeze(-1)
        else:
            x = self.prelu(x)
        return x

def export_coreml_mtcnn(mtcnn, save_dir):
    import coremltools as ct
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device('cpu')
    mtcnn.to(device)
    mtcnn.eval()

    # --- Apply PReLU fix for RNet and ONet ---
    # RNet: prelu4 follows dense4 (Linear)
    if hasattr(mtcnn.rnet, 'prelu4') and not isinstance(mtcnn.rnet.prelu4, Rank4PReLU):
        print("Patching RNet.prelu4 for CoreML compatibility...")
        mtcnn.rnet.prelu4 = Rank4PReLU(mtcnn.rnet.prelu4)

    # ONet: prelu5 follows dense5 (Linear)
    if hasattr(mtcnn.onet, 'prelu5') and not isinstance(mtcnn.onet.prelu5, Rank4PReLU):
        print("Patching ONet.prelu5 for CoreML compatibility...")
        mtcnn.onet.prelu5 = Rank4PReLU(mtcnn.onet.prelu5)

    
    # --- Export PNet ---
    print("Exporting PNet...")
    # PNet is FCN. Input (1, 3, H, W). Output (1, 4, H', W'), (1, 2, H', W')
    # We use flexible input shape for H and W.
    pnet = mtcnn.pnet
    dummy_input_pnet = torch.randn(1, 3, 100, 100) # Arbitrary size
    traced_pnet = torch.jit.trace(pnet, dummy_input_pnet)
    
    # Define flexible shape for PNet
    # RangeDim for H and W.
    input_shape_pnet = ct.Shape(shape=(1, 3, ct.RangeDim(12, 1024), ct.RangeDim(12, 1024)))
    
    # Important: PNet returns (offsets, probs). 
    # Let's name outputs explicitly to help the wrapper logic.
    # Actually `torch.jit.trace` preserves order.
    # converted model output names: var_xx. 
    # We can rename propertie output_names=['offsets', 'probs']
    
    model_pnet = ct.convert(
        traced_pnet,
        inputs=[ct.TensorType(name="input_1", shape=input_shape_pnet)],
        outputs=[ct.TensorType(name="output_1"), ct.TensorType(name="output_2")], # Force order?
        compute_precision=ct.precision.FLOAT32
    )
    # Note: convert might not guarantee output order mapping to "output_1", "output_2" 
    # unless we check the source graph. 
    # But usually it follows return order.
    model_pnet.save(os.path.join(save_dir, "PNet.mlpackage"))
    
    # --- Export RNet ---
    print("Exporting RNet...")
    rnet = mtcnn.rnet
    # RNet input is fixed 24x24. Batch size can be flexible.
    dummy_input_rnet = torch.randn(1, 3, 24, 24)
    traced_rnet = torch.jit.trace(rnet, dummy_input_rnet)
    
    # Flexible batch size
    input_shape_rnet = ct.Shape(shape=(ct.RangeDim(1, 100), 3, 24, 24))
    
    model_rnet = ct.convert(
        traced_rnet,
        inputs=[ct.TensorType(name="input_1", shape=input_shape_rnet)],
        outputs=[ct.TensorType(name="output_1"), ct.TensorType(name="output_2")],
        compute_precision=ct.precision.FLOAT32
    )
    model_rnet.save(os.path.join(save_dir, "RNet.mlpackage"))

    # --- Export ONet ---
    print("Exporting ONet...")
    onet = mtcnn.onet
    # ONet input is fixed 48x48.
    dummy_input_onet = torch.randn(1, 3, 48, 48)
    traced_onet = torch.jit.trace(onet, dummy_input_onet)
    
    input_shape_onet = ct.Shape(shape=(ct.RangeDim(1, 100), 3, 48, 48))
    
    # ONet returns (offsets, probs, points) -> 3 outputs
    model_onet = ct.convert(
        traced_onet,
        inputs=[ct.TensorType(name="input_1", shape=input_shape_onet)],
        outputs=[ct.TensorType(name="output_1"), ct.TensorType(name="output_2"), ct.TensorType(name="output_3")],
        compute_precision=ct.precision.FLOAT32
    )
    model_onet.save(os.path.join(save_dir, "ONet.mlpackage"))
    
    print(f"Export completed. Models saved to {save_dir}")

# ------------------------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to input image")
    parser.add_argument("--det_max_side", type=int, default=320, help="Max side for resizing (default 320)")
    parser.add_argument("--export", action="store_true", help="Export PNet, RNet, ONet to CoreML")
    parser.add_argument("--onnx", action="store_true", help="Use ONNX models for inference")
    parser.add_argument("--coreml", action="store_true", help="Use CoreML models for inference")
    parser.add_argument("--models_dir", type=str, default="detection_models", help="Directory for models")
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device} (Main control flow)")

    # Initialize MTCNN
    # keep_all=True so we detect all faces
    mtcnn = MTCNN(keep_all=True, device=device)
    
    if args.export:
        print("Export mode selected.")
        export_coreml_mtcnn(mtcnn, args.models_dir)
        return

    if args.input is None:
        print("Please provide --input <image_path>")
        return

    # Load Image
    img_rgb = load_rgb_uint8(args.input)
    img_resized, scale = resize_by_max_side(img_rgb, args.det_max_side)
    print(f"Image loaded: {args.input}, Original: {img_rgb.shape}, Resized: {img_resized.shape}, Scale: {scale:.4f}")

    # Inject Inference Backends if needed
    if args.onnx:
        print("Switching to ONNX backend...")
        # Assuming ONNX models exist in args.models_dir or default relative location
        # As per onnx_baseline.py instructions, they are expected in 'onnx_models' usually
        # But for this test, let's assume user provides them or we look in typical place.
        # onnx_baseline.py looked in "onnx_models" sibling to script.
        # We will check args.models_dir first.
        
        pnet_path = os.path.join(args.models_dir, "mtcnn_pnet.onnx")
        rnet_path = os.path.join(args.models_dir, "mtcnn_rnet.onnx")
        onet_path = os.path.join(args.models_dir, "mtcnn_onet.onnx")
        
        if not all(os.path.exists(p) for p in [pnet_path, rnet_path, onet_path]):
            print(f"[ERROR] ONNX models not found in {args.models_dir}. Please ensure they exist.")
            return

        mtcnn.pnet = GenericONNXModel(pnet_path)
        mtcnn.rnet = GenericONNXModel(rnet_path)
        mtcnn.onet = GenericONNXModel(onet_path)

    elif args.coreml:
        print("Switching to CoreML backend...")
        pnet_path = os.path.join(args.models_dir, "PNet.mlpackage")
        rnet_path = os.path.join(args.models_dir, "RNet.mlpackage")
        onet_path = os.path.join(args.models_dir, "ONet.mlpackage")
        
        if not all(os.path.exists(p) for p in [pnet_path, rnet_path, onet_path]):
            print(f"[ERROR] CoreML models not found in {args.models_dir}. Run --export first.")
            return

        mtcnn.pnet = GenericCoreMLModel(pnet_path)
        mtcnn.rnet = GenericCoreMLModel(rnet_path)
        mtcnn.onet = GenericCoreMLModel(onet_path)

    else:
        print("Using default PyTorch backend.")

    # Run Detection
    t0 = time.time()
    # mtcnn.detect returns boxes (N, 4) and probs (N,)
    # If landmarks=True, it returns points (N, 5, 2)
    boxes, probs, points = mtcnn.detect(img_resized, landmarks=True)
    t1 = time.time()
    
    print(f"Detection took: {(t1 - t0)*1000:.2f} ms")
    
    if boxes is not None:
        print(f"Found {len(boxes)} faces.")
        for i, box in enumerate(boxes):
            # Scale box back to original
            box_orig = box / scale
            print(f" Face {i+1}: {box_orig.astype(int)} | Prob: {probs[i]:.4f}")
    else:
        print("No faces found.")

if __name__ == "__main__":
    main()
