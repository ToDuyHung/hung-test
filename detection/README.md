# Face Detection Test & Export Tool

This directory contains a script `test.py` to evaluate MTCNN face detection using various backends (PyTorch, ONNX, CoreML) and to export the PyTorch models to CoreML.

## Usage

### 1. Export CoreML Models
First, generate the `.mlpackage` files. This will save `PNet`, `RNet`, and `ONet` to the specified directory.
```bash
python test.py --export --models_dir coreml_models
```

### 2. Run Detection (PyTorch - Default)
Run standard PyTorch inference (CPU/GPU auto-detected).
```bash
python test.py --input P1E_S1_C1_00001452.jpg
```

### 3. Run Detection (ONNX)
Run inference using ONNX Runtime. Requires `onnx_models` directory to contain `mtcnn_pnet.onnx`, `mtcnn_rnet.onnx`, `mtcnn_onet.onnx`.
```bash
python hung-test/detection/test.py --input <path_to_image> --onnx --models_dir onnx_models
```

### 4. Run Detection (CoreML)
Run inference using CoreML. Requires Step 1 to be completed.
```bash
python test.py --input P1E_S1_C1_00001452.jpg --coreml --models_dir hung-test/detection/coreml_models
```

## Options
- `--det_max_side`: Max side length to resize the input image (maintain aspect ratio). Default: `320`.
- `--models_dir`: Directory to load/save models. Default: `detection_models`.

---

## Swift Performance Optimization

### Problem
Swift CoreML inference was running at **~500ms** vs Python's **~50ms** (10x slower).

### Root Causes & Fixes

#### 1. 🚀 **CIContext Recreation** (Critical)
**Problem**: Creating new `CIContext()` for every image/crop (~50-100 times per frame)  
**Impact**: ~300-400ms overhead  
**Fix**: Reuse single `ciContext` instance created in `init()`

```swift
// Before
let context = CIContext()
context.render(image, to: buffer)

// After
private let ciContext: CIContext // Created once
ciContext.render(image, to: buffer)
```

#### 2. ⚡ **MLModelConfiguration** (High Priority)
**Problem**: Default config doesn't utilize Neural Engine  
**Impact**: ~50-100ms overhead  
**Fix**: Set `computeUnits = .all`

```swift
let config = MLModelConfiguration()
config.computeUnits = .all // Neural Engine + GPU + CPU
```

#### 3. 📊 **Float16 Extraction** (Medium Priority)
**Problem**: Slow subscript access for non-Float32 arrays  
**Impact**: ~10-20ms overhead  
**Fix**: Added fast unsafe pointer path for Float16

### Expected Performance
- **Before**: ~500ms per frame
- **After**: ~50-100ms per frame (5-10x speedup)

### Benchmarking
```swift
let start = CFAbsoluteTimeGetCurrent()
let results = try await detector.detectFaces(image: image)
let elapsed = (CFAbsoluteTimeGetCurrent() - start) * 1000
print("Detection took: \(elapsed)ms")
```

