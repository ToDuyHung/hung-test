#!/usr/bin/env python3
"""
Test script for head pose estimation with CoreML export/inference support.
"""
import argparse
import os
import sys

# Add parent directory to path to import head_pose module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from head_pose import PoseAngleValidator6DRepNet
import cv2
import time


def main():
    parser = argparse.ArgumentParser(description="Head Pose Estimation Test")
    parser.add_argument("--input", type=str, help="Path to input image")
    parser.add_argument("--export", action="store_true", help="Export model to CoreML")
    parser.add_argument("--coreml", action="store_true", help="Use CoreML for inference")
    parser.add_argument("--model_path", type=str, default="SixDRepNet.mlpackage", 
                        help="Path to CoreML model (for --coreml) or export path (for --export)")
    parser.add_argument("--weights", type=str, default="", 
                        help="Path to PyTorch weights (empty = download from URL)")
    parser.add_argument("--device", type=int, default=-1, 
                        help="GPU device ID (-1 for CPU)")
    
    args = parser.parse_args()
    
    # Initialize validator
    print(f"Initializing PoseAngleValidator6DRepNet (device={args.device})...")
    validator = PoseAngleValidator6DRepNet(device=args.device, dict_path=args.weights)
    
    # Export mode
    if args.export:
        print("Export mode selected.")
        validator.export_coreml(args.model_path)
        return
    
    # Load CoreML if requested
    if args.coreml:
        if not os.path.exists(args.model_path):
            print(f"[ERROR] CoreML model not found at {args.model_path}. Run --export first.")
            return
        validator.load_coreml(args.model_path)
        print("Using CoreML backend.")
    else:
        print("Using PyTorch backend.")
    
    # Inference mode
    if args.input is None:
        print("Please provide --input <image_path>")
        return
    
    # Load image
    img = cv2.imread(args.input)
    if img is None:
        print(f"[ERROR] Failed to load image: {args.input}")
        return
    
    print(f"Image loaded: {args.input}, Shape: {img.shape}")
    
    # Run inference
    t0 = time.time()
    result = validator.check_image(img)
    t1 = time.time()
    
    print(f"Inference took: {(t1 - t0)*1000:.2f} ms")
    print(f"Result: {result}")
    
    if result['pose']:
        pose = result['pose']
        print(f"  Pitch: {pose['pitch']:.2f}°")
        print(f"  Yaw:   {pose['yaw']:.2f}°")
        print(f"  Roll:  {pose['roll']:.2f}°")
        print(f"  View:  {result['view']}")


if __name__ == "__main__":
    main()
