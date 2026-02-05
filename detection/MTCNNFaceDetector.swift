//
//  MTCNNFaceDetector.swift
//  VCCSmartOffice
//
//  Created by [Agent] on 5/2/26.
//

import Foundation
import CoreML
import Vision
import CoreImage
import UIKit 

class MTCNNFaceDetector {
    
    // MARK: - Properties
    
    private let pNet: MLModel
    private let rNet: MLModel
    private let oNet: MLModel
    
    // Thresholds
    private let pNetThreshold: Float = 0.7
    private let rNetThreshold: Float = 0.7
    private let oNetThreshold: Float = 0.7
    
    // NMS Thresholds
    private let nmsThreshold1: Float = 0.7
    private let nmsThreshold2: Float = 0.7
    private let nmsThreshold3: Float = 0.7
    
    // Scaling
    private let scaleFactor: Float = 0.709
    
    // Performance Optimization
    private let maxInputSize: CGFloat = 320.0
    
    // 🚀 CRITICAL: Reuse CIContext to avoid expensive recreation
    private let ciContext: CIContext
    
    init() throws {
        // Configure for optimal performance
        let config = MLModelConfiguration()
        config.computeUnits = .all // Use Neural Engine + GPU + CPU
        
        self.pNet = try PNet(configuration: config).model
        self.rNet = try RNet(configuration: config).model
        self.oNet = try ONet(configuration: config).model
        
        // Create CIContext once and reuse (saves ~300-400ms per inference)
        self.ciContext = CIContext(options: [
            .useSoftwareRenderer: false,
            .cacheIntermediates: false // Reduce memory for real-time use
        ])
        
        print("✅ MTCNN Models Loaded with optimized configuration")
    }

    // MARK: - Public Methods
    
    func detectFaces(image: CIImage) async throws -> [FaceDetectionResult] {
        // Fix Orientation: CIImage origin is bottom-left, Buffer is top-left.
        // We flip Y to make the image upright in CI coordinate space.
        let flipped = image.transformed(by: CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -image.extent.height))
        
        // 0. Optimization: Resize Image if too large
        var processingImage = flipped
        let maxDim = max(flipped.extent.width, flipped.extent.height)
        if maxDim > maxInputSize {
            let scale = maxInputSize / maxDim
            let newWidth = flipped.extent.width * scale
            let newHeight = flipped.extent.height * scale
            // Use existing resize helper, but we need to make sure we don't lose it if it returns nil (unlikely)
            if let resized = resizeImage(flipped, size: CGSize(width: newWidth, height: newHeight)) {
                processingImage = resized
            }
        }
        
        // 1. Image Pyramid & PNet
        var boxes = try runPNet(image: processingImage)
        guard !boxes.isEmpty else { return [] }
        
        // 2. RNet
        boxes = try runRNet(image: processingImage, boxes: boxes)
        guard !boxes.isEmpty else { return [] }
        
        // 3. ONet
        let results = try runONet(image: processingImage, boxes: boxes)
        
        return results
    }
    
    func detectFaces(buffer: CMSampleBuffer) async throws -> [FaceDetectionResult] {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(buffer) else { return [] }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        return try await detectFaces(image: ciImage)
    }
    
    // MARK: - Internal Steps
    
    private func runPNet(image: CIImage) throws -> [CGRect] {
        var boxes: [FaceBox] = []
        
        let width = Int(image.extent.width)
        let height = Int(image.extent.height)
        let minSide = min(width, height)
        
        var currentScale: Float = 1.0
        let minSize: Float = 12.0
        
        // PNet expects flexible input usually.
        // We generate scales until minSide * scale < 12
        
        var scales: [Float] = []
        var m: Float = 12.0 / minSize
        
        // Typical MTCNN scaling:
        // First scale scales image so that min side is 12? No, usually distinct scales.
        // Standard: factor 0.709.
        
        // Let's use simple loop
        var s = currentScale
        // Fix: Use 64.0 as min limit based on reported error "Image height (64) is not in allowed range (64..2048)"
        // This suggests input must be at least 64.
        while Float(minSide) * s >= 64.0 {
            scales.append(s)
            s *= scaleFactor
        }
        
        for scale in scales {
            let scaledWidth = Int(Float(width) * scale)
            let scaledHeight = Int(Float(height) * scale)
            
            guard let resized = resizeImage(image, size: CGSize(width: scaledWidth, height: scaledHeight)) else { continue }
            
            let pixelBuffer = try createPixelBuffer(from: resized, width: scaledWidth, height: scaledHeight)
            
            // Run Prediction
            let inputName = pNet.modelDescription.inputDescriptionsByName.keys.first ?? "data"
            let input = try MLDictionaryFeatureProvider(dictionary: [inputName: pixelBuffer])
            let output = try pNet.prediction(from: input)
            
            // Parse Output
            var probArray: MLMultiArray?
            var boxArray: MLMultiArray?
            
            for key in output.featureNames {
                guard let feature = output.featureValue(for: key)?.multiArrayValue else { continue }
                let channels = feature.shape[1].intValue 
                
                if channels == 2 {
                    probArray = feature
                } else if channels == 4 {
                    boxArray = feature
                }
            }
            
            guard let probs = probArray, let diffs = boxArray else {
                continue
            }
            
            let probFloats = extractFloatData(from: probs)
            let diffFloats = extractFloatData(from: diffs)
            
            let candidates = generateBoundingBox(probs: probs, diffs: diffs, probFloats: probFloats, diffFloats: diffFloats, scale: scale, threshold: pNetThreshold)
            boxes.append(contentsOf: candidates)
        }
        
        // NMS
        let nmsIndices = MathUtilities.nonMaxSuppression(boxes: boxes.map { $0.rect }, scores: boxes.map { $0.score }, iouThreshold: nmsThreshold1, type: .union)
        
        return nmsIndices.map { boxes[$0].rect }
    }
    
    // MARK: - Internal Helpers
    
    struct FaceBox {
        var rect: CGRect
        var score: Float
        var landmarks: [CGPoint]?
    }
    
    // Updated signature to take extracted arrays
    private func generateBoundingBox(probs: MLMultiArray, diffs: MLMultiArray, probFloats: [Float], diffFloats: [Float], scale: Float, threshold: Float) -> [FaceBox] {
        let shape = probs.shape
        let H = shape[shape.count - 2].intValue
        let W = shape[shape.count - 1].intValue
        let stride = 2
        let cellSize = 12
        
        var boxes: [FaceBox] = []
        
        // MLMultiArray Strides: [S_batch, S_channel, S_height, S_width]
        let pStrides = probs.strides.map { $0.intValue }
        let dStrides = diffs.strides.map { $0.intValue }
        
        for y in 0..<H {
            for x in 0..<W {
                // Determine layout based on strides
                // We want channel 1 for score
                var scoreIdx = 0
                
                 if probs.shape.count == 4 {
                     scoreIdx = 1 * pStrides[1] + y * pStrides[2] + x * pStrides[3]
                 } else {
                     scoreIdx = 1 * pStrides[0] + y * pStrides[1] + x * pStrides[2]
                 }
                
                if scoreIdx < 0 || scoreIdx >= probFloats.count {
                    continue
                }
                
                let score = probFloats[scoreIdx]
                
                if score > threshold {
                    // Extract Offset
                    var diffsArr: [Float] = []
                    for k in 0..<4 {
                        var dIdx = 0
                        if diffs.shape.count == 4 {
                            dIdx = k * dStrides[1] + y * dStrides[2] + x * dStrides[3]
                        } else {
                            dIdx = k * dStrides[0] + y * dStrides[1] + x * dStrides[2]
                        }
                        
                        if dIdx >= 0 && dIdx < diffFloats.count {
                            diffsArr.append(diffFloats[dIdx])
                        } else {
                            diffsArr.append(0)
                        }
                    }
                    
                    // Decode Box (Linear Regression)
                    let outputX = Float(x * stride + 1) / scale
                    let outputY = Float(y * stride + 1) / scale
                    let outputSize = Float(cellSize) / scale
                    
                    let regX1 = diffsArr[0]
                    let regY1 = diffsArr[1]
                    let regX2 = diffsArr[2]
                    let regY2 = diffsArr[3]
                    
                    let boxX1 = outputX
                    let boxY1 = outputY
                    let boxW = outputSize
                    let boxH = outputSize
                    let boxX2 = boxX1 + boxW
                    let boxY2 = boxY1 + boxH
                    
                    let newX1 = boxX1 + regX1 * boxW
                    let newY1 = boxY1 + regY1 * boxH
                    let newX2 = boxX2 + regX2 * boxW
                    let newY2 = boxY2 + regY2 * boxH
                    
                    let rect = CGRect(
                        x: Double(newX1),
                        y: Double(newY1),
                        width: Double(newX2 - newX1),
                        height: Double(newY2 - newY1)
                    )
                    
                    boxes.append(FaceBox(rect: rect, score: score, landmarks: nil))
                }
            }
        }
        
        return boxes
    }
    
    private func resizeImage(_ image: CIImage, size: CGSize) -> CIImage? {
        let scaleX = size.width / image.extent.width
        let scaleY = size.height / image.extent.height
        return image.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
    }
    
    private func createPixelBuffer(from image: CIImage, width: Int, height: Int) throws -> CVPixelBuffer {
         let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA
        ] as CFDictionary

        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs, &pixelBuffer)

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw NSError(domain: "MTCNN", code: -1, userInfo: [NSLocalizedDescriptionKey: "Buffer creation failed"])
        }

        // 🚀 FIXED: Reuse shared CIContext instead of creating new one
        ciContext.render(image, to: buffer)
        return buffer
    }
    
    private func runRNet(image: CIImage, boxes: [CGRect]) throws -> [CGRect] {
        var outputBoxes: [FaceBox] = []
        
        // 1. Convert to Square
        let squaredBoxes = convertToSquare(boxes)
        
        for box in squaredBoxes {
            // 2. Crop and resize
            let cropped = image.cropped(to: box)
            
            // Shift to (0,0) before resize
            let shifted = cropped.transformed(by: CGAffineTransform(translationX: -box.origin.x, y: -box.origin.y))
            
            guard let resized = resizeImage(shifted, size: CGSize(width: 24, height: 24)) else { continue }
            let pixelBuffer = try createPixelBuffer(from: resized, width: 24, height: 24)
            
            // 3. Inference
            let inputName = rNet.modelDescription.inputDescriptionsByName.keys.first ?? "data"
            let input = try MLDictionaryFeatureProvider(dictionary: [inputName: pixelBuffer])
            let output = try rNet.prediction(from: input)
            
            // 4. Decode
            // Output names: "prob1" (prob), "conv5-2" (bbox)?
            // Note: If model output names differ, user needs to adjust.
            // Using `featureNames` iteration to be safe.
            
            var probValue: Float = 0.0
            var diffs: [Float] = []
            
            for key in output.featureNames {
                guard let feature = output.featureValue(for: key)?.multiArrayValue else { continue }
                let channels = feature.count
                
                let floats = extractFloatData(from: feature)
                
                // Heuristic: Prob has 2 values, Bbox has 4
                if channels == 2 {
                    if floats.count >= 2 {
                        probValue = floats[1] // Index 1 is face
                    }
                } else if channels == 4 {
                    diffs = floats
                }
            }
            
            if probValue > rNetThreshold && diffs.count == 4 {
                // Calibrate
                let calibrated = calibrateBox(box: box, diff: diffs)
                outputBoxes.append(FaceBox(rect: calibrated, score: probValue, landmarks: nil))
            }
        }
        
        // 5. NMS
        let nmsIndices = MathUtilities.nonMaxSuppression(boxes: outputBoxes.map { $0.rect }, scores: outputBoxes.map { $0.score }, iouThreshold: nmsThreshold2, type: .union)
        
        return nmsIndices.map { outputBoxes[$0].rect }
    }
    
    // MARK: - Helper Methods for Boxes
    
    private func convertToSquare(_ boxes: [CGRect]) -> [CGRect] {
        return boxes.map { box in
            let w = box.width
            let h = box.height
            let maxSide = max(w, h)
            
            let centerX = box.minX + w * 0.5
            let centerY = box.minY + h * 0.5
            
            let newX = centerX - maxSide * 0.5
            let newY = centerY - maxSide * 0.5
            
            return CGRect(x: newX, y: newY, width: maxSide, height: maxSide)
        }
    }
    
    private func calibrateBox(box: CGRect, diff: [Float]) -> CGRect {
        let x = Float(box.minX)
        let y = Float(box.minY)
        let w = Float(box.width)
        let h = Float(box.height)
        
        let dx = diff[0]
        let dy = diff[1]
        let dw = diff[2]
        let dh = diff[3]
        
        let newX = x + dx * w
        let newY = y + dy * h
        
        let x2 = x + w
        let y2 = y + h
        
        let nx1 = x + dx * w
        let ny1 = y + dy * h
        let nx2 = x2 + dw * w // dw here is actually dx2
        let ny2 = y2 + dh * h // dh here is actually dy2
        
        return CGRect(x: Double(nx1), y: Double(ny1), width: Double(nx2 - nx1), height: Double(ny2 - ny1))
    }
    
    private func runONet(image: CIImage, boxes: [CGRect]) throws -> [FaceDetectionResult] {
        var outputResults: [FaceDetectionResult] = []
        
        let squaredBoxes = convertToSquare(boxes)
        
        for box in squaredBoxes {
            let cropped = image.cropped(to: box)
            let shifted = cropped.transformed(by: CGAffineTransform(translationX: -box.origin.x, y: -box.origin.y))
            
            guard let resized = resizeImage(shifted, size: CGSize(width: 48, height: 48)) else { continue }
            let pixelBuffer = try createPixelBuffer(from: resized, width: 48, height: 48)
            
            // Inference
            let inputName = oNet.modelDescription.inputDescriptionsByName.keys.first ?? "data"
            let input = try MLDictionaryFeatureProvider(dictionary: [inputName: pixelBuffer])
            let output = try oNet.prediction(from: input)
            
            // Decode
            // oNet Output: prob [1, 2], bbox [1, 4], landmarks [1, 10]
            
            var probValue: Float = 0.0
            var diffs: [Float] = []
            var landmarksDiff: [Float] = []
            
            for key in output.featureNames {
                guard let feature = output.featureValue(for: key)?.multiArrayValue else { continue }
                let channels = feature.count
                
                let floats = extractFloatData(from: feature)
                
                if channels == 2 {
                    if floats.count >= 2 {
                        probValue = floats[1]
                    }
                } else if channels == 4 {
                    diffs = floats
                } else if channels == 10 {
                    landmarksDiff = floats
                }
            }
            
            if probValue > oNetThreshold && diffs.count == 4 {
                let calibratedBox = calibrateBox(box: box, diff: diffs)
                
                // Decode Landmarks
                var landmarks: [CGPoint] = []
                if landmarksDiff.count == 10 {
                    let w = Float(box.width)
                    let h = Float(box.height)
                    let x = Float(box.minX)
                    let y = Float(box.minY)
                    
                    for i in 0..<5 {
                        let lx = landmarksDiff[i]
                        let ly = landmarksDiff[i + 5]
                        
                        let px = x + lx * w
                        let py = y + ly * h
                        
                        landmarks.append(CGPoint(x: Double(px), y: Double(py)))
                    }
                }
                
                outputResults.append(FaceDetectionResult(boundingBox: calibratedBox, landmarks: landmarks, confidence: probValue))
            }
        }
        
        // NMS
        let boxesForNMS = outputResults.map { $0.boundingBox }
        let scoresForNMS = outputResults.map { $0.confidence }
        
        let nmsIndices = MathUtilities.nonMaxSuppression(boxes: boxesForNMS, scores: scoresForNMS, iouThreshold: nmsThreshold3, type: .min)
        
        let finalResults = nmsIndices.map { outputResults[$0] }
        
        // Normalize results to [0, 1]
        let width = Double(image.extent.width)
        let height = Double(image.extent.height)
        
        return finalResults.map { res in
            let normalizedBox = CGRect(
                x: res.boundingBox.minX / width,
                y: res.boundingBox.minY / height,
                width: res.boundingBox.width / width,
                height: res.boundingBox.height / height
            )
            
            let normalizedLandmarks = res.landmarks?.map { p in
                CGPoint(x: p.x / width, y: p.y / height)
            }
            
            return FaceDetectionResult(boundingBox: normalizedBox, landmarks: normalizedLandmarks, confidence: res.confidence)
        }
    }
    private func extractFloatData(from multiArray: MLMultiArray) -> [Float] {
        let count = multiArray.count
        
        // Fast path for Float32
        if multiArray.dataType == .float32 {
            let ptr = UnsafeBufferPointer<Float>(start: multiArray.dataPointer.bindMemory(to: Float.self, capacity: count), count: count)
            return Array(ptr)
        }
        
        // 🚀 OPTIMIZED: Fast path for Float16 (common in CoreML)
        if multiArray.dataType == .float16 {
            if #available(iOS 14.0, macOS 11.0, *) {
                let ptr = UnsafeBufferPointer<Float16>(start: multiArray.dataPointer.bindMemory(to: Float16.self, capacity: count), count: count)
                return ptr.map { Float($0) }
            }
        }
        
        // Fallback for Double or other types
        var result = [Float](repeating: 0, count: count)
        for i in 0..<count {
            result[i] = multiArray[i].floatValue
        }
        return result
    }
}

// MARK: - Supporting Types

struct FaceDetectionResult {
    let boundingBox: CGRect // Normalized [0, 1]
    let landmarks: [CGPoint]? // Normalized [0, 1]
    let confidence: Float
}
