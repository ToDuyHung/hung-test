//
//  MathUtilities.swift
//  VCCSmartOffice
//
//  Created by Leo on 9/12/25.
//

import Foundation
import Accelerate

enum MathUtilities {
    static func calculateCosineSimilarity(_ u: [Float], _ v: [Float]) -> Float {
        // 1. Check if dimensions match
        guard u.count == v.count, u.count > 0 else { return 0 }
        
        // 2. Calculate the dot product of u and v
        let dotProduct = vDSP.dot(u, v)
        
        // 3. Calculate the magnitude (L2 norm) of u and v
        // Note: vDSP.dot(u, u) calculates the sum of squares directly
        let magU = sqrt(vDSP.dot(u, u))
        let magV = sqrt(vDSP.dot(v, v))
        
        // 4. Handle division by zero (if a vector is all zeros)
        if magU == 0 || magV == 0 { return 0.0 }
        
        // 5. Calculate Cosine Similarity
        return dotProduct / (magU * magV)
    }
    
    static func normalize(_ vector: [Float]) -> [Float] {
        let count = vDSP_Length(vector.count)
        var sumSq: Float = 0.0
        vDSP_svesq(vector, 1, &sumSq, count)
        
        let norm = sqrt(sumSq)
        let epsilon: Float = 1e-12
        
        guard norm > epsilon else {
            return vector
        }
        
        // Multiply by 1/norm
        var result = [Float](repeating: 0.0, count: vector.count)
        var scale = 1.0 / norm
        vDSP_vsmul(vector, 1, &scale, &result, 1, count)
        
        return result
    }
    
    // MARK: - Bounding Box Utilities
    
    /// Calculates Intersection over Union (IoU) between two rects
    static func intersectionOverUnion(_ boxA: CGRect, _ boxB: CGRect) -> Float {
        let xA = max(boxA.minX, boxB.minX)
        let yA = max(boxA.minY, boxB.minY)
        let xB = min(boxA.maxX, boxB.maxX)
        let yB = min(boxA.maxY, boxB.maxY)
        
        let interWidth = max(0, xB - xA)
        let interHeight = max(0, yB - yA)
        
        // Compute the area of intersection rectangle
        let interArea = interWidth * interHeight
        
        if interArea == 0 { return 0.0 }
        
        let boxAArea = boxA.width * boxA.height
        let boxBArea = boxB.width * boxB.height
        
        // Compute the area of both rectangles combined (union)
        let unionArea = boxAArea + boxBArea - interArea
        
        return Float(interArea / unionArea)
    }
    
    /// Non-Maximum Suppression (NMS)
    static func nonMaxSuppression(boxes: [CGRect], scores: [Float], iouThreshold: Float, type: NMSType = .union) -> [Int] {
        guard !boxes.isEmpty, boxes.count == scores.count else { return [] }
        
        // Sort indices by score (descending)
        let sortedIndices = scores.enumerated()
            .sorted(by: { $0.element > $1.element })
            .map { $0.offset }
        
        var selectedIndices: [Int] = []
        var activeIndices = sortedIndices
        
        while !activeIndices.isEmpty {
            let current = activeIndices.removeFirst()
            selectedIndices.append(current)
            
            let currentBox = boxes[current]
            
            activeIndices = activeIndices.filter { index in
                let box = boxes[index]
                let iou = intersectionOverUnion(currentBox, box)
                
                // If IoU is greater than threshold, it overlaps too much, so suppress it (filter it out)
                return iou <= iouThreshold
            }
        }
        
        return selectedIndices
    }
    
    enum NMSType {
        case union
        case min
    }
}
