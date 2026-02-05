# Head Pose Estimation Test & Export Tool

This directory contains scripts for head pose estimation using SixDRepNet (6D rotation representation network).

## Usage

### 1. Export CoreML Model
Generate the `.mlpackage` file for CoreML deployment.
```bash
python test.py --export --model_path SixDRepNet.mlpackage
```

### 2. Run Inference (PyTorch - Default)
Run standard PyTorch inference (CPU/GPU auto-detected).
```bash
python test.py --input <path_to_face_crop.jpg>
```

### 3. Run Inference (CoreML)
Run inference using CoreML model (macOS only). Requires Step 1 to be completed.
```bash
python test.py --input <path_to_face_crop.jpg> --coreml --model_path SixDRepNet.mlpackage
```

## Options
- `--device`: GPU device ID (-1 for CPU). Default: `-1`.
- `--weights`: Path to PyTorch weights (empty = auto-download). Default: `""`.
- `--model_path`: Path for CoreML model. Default: `SixDRepNet.mlpackage`.

## Model Details

### SixDRepNet Architecture
- **Backbone**: RepVGG-B1g2 (deployed mode)
- **Input**: 224×224 RGB image (face crop)
- **Output**: 3×3 rotation matrix
- **Euler Angles**: Computed from rotation matrix (pitch, yaw, roll in degrees)

### Head Pose Classification
The model classifies head pose into 5 views:
- **Front**: |yaw| ≤ 15°, |pitch| ≤ 12°
- **Left**: yaw ∈ [-60°, -20°]
- **Right**: yaw ∈ [20°, 60°]
- **Up**: pitch ∈ [10°, 45°]
- **Down**: pitch ∈ [-45°, -10°]

Additional gate: |roll| ≤ 25° (reject if too tilted)

## Example Output
```
Image loaded: face_crop.jpg, Shape: (224, 224, 3)
Inference took: 15.32 ms
Result: {'view': 'front', 'pose': {'pitch': 5.2, 'yaw': -3.1, 'roll': 1.8}, 'pass_map': {...}}
  Pitch: 5.20°
  Yaw:   -3.10°
  Roll:  1.80°
  View:  front
```

## Notes
- CoreML inference only works on macOS (10.13+)
- On Linux/WSL, use PyTorch backend only
- Face crop should be centered and properly aligned for best results
