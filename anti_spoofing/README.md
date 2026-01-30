# Face Anti-Spoofing Module

This directory contains a PyTorch-based face anti-spoofing pipeline using **MiniFASNetV2**. It is designed for lightweight, mobile-friendly deployment and supports CoreML export.

## Features

- **Model**: MiniFASNetV2 (from [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)).
- **Preprocessing**: Official "Scale 2.7" cropping logic to ensure high accuracy.
- **Export**: Built-in support to convert the model to iOS `.mlmodel`.
- **Performance**: High accuracy (~99% on sample tests) with low computational cost (0.43M params).

## Structure

- `pytorch_baseline.py`: Main script for inference and CoreML export.
- `src/`: Helper libraries (ported from official repo) for model architecture and data preprocessing.
- `2.7_80x80_MiniFASNetV2.pth`: Pretrained model weights (download required if missing).
- `test_images/`: Sample images for benchmarking.

## Setup

1.  **Environment**:
    ```bash
    conda activate facenet
    pip install coremltools  # For iOS export
    ```

2.  **Download Weights**:
    If not already present, download the weights:
    ```bash
    wget -O 2.7_80x80_MiniFASNetV2.pth "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
    ```

## Usage

### Run Inference
Run various images in a directory:
```bash
python pytorch_baseline.py --data_root ./test_images --model_path 2.7_80x80_MiniFASNetV2.pth
```

### Export to CoreML
Generate `MiniFASNetV2.mlmodel` for iOS:
```bash
python pytorch_baseline.py --export_coreml --model_path 2.7_80x80_MiniFASNetV2.pth
```

## Benchmark Results
Verified against official sample images using the integrated official preprocessing logic:

| Image | Label | Score (Confidence) | Result |
|---|---|---|---|
| `image_T1.jpg` | **Real** | **0.9999** | PASSED |
| `image_F1.jpg` | **Spoof** | 0.9872 | PASSED |

*Note: The pipeline uses MTCNN for face detection, but automatically applies the required aspect-ratio corrections and scaling (2.7x) to match the MiniFASNet training distribution.*
