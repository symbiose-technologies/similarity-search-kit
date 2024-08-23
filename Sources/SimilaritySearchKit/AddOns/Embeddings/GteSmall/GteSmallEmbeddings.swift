//////////////////////////////////////////////////////////////////////////////////
//
//  SYMBIOSE
//  Copyright 2023 Symbiose Technologies, Inc
//  All Rights Reserved.
//
//  NOTICE: This software is proprietary information.
//  Unauthorized use is prohibited.
//
// 
// Created by: Ryan Mckinney on 8/8/23
//
////////////////////////////////////////////////////////////////////////////////

import Foundation
import CoreML
import SimilaritySearchKit

@available(macOS 13.0, iOS 16.0, *)
public class GteSmallEmbeddings: EmbeddingsProtocol {

    public let model: thenlper_gte_small

//    #if os(macOS)
//    public let model: thenlper_gte_small
//    #else
//    public let model: thenlper_gte_small_quantized_8_bit
//    #endif
    public let tokenizer: BertTokenizer
    public let inputDimention: Int = 512
    public let outputDimention: Int = 384

    
    public init() {
        let modelConfig = MLModelConfiguration()
       
        modelConfig.computeUnits = .all
        do {
            self.model = try thenlper_gte_small(configuration: modelConfig)
        } catch {
            fatalError("Failed to load the Core ML model. Error: \(error.localizedDescription)")
        }
        
//        #if os(macOS)
//        modelConfig.computeUnits = .all
//        do {
//            self.model = try thenlper_gte_small(configuration: modelConfig)
//        } catch {
//            fatalError("Failed to load the Core ML model. Error: \(error.localizedDescription)")
//        }
//        #else
//        modelConfig.computeUnits = .cpuAndGPU
//        do {
//            self.model = try thenlper_gte_small_quantized_8_bit(configuration: modelConfig)
//        } catch {
//            fatalError("Failed to load the Core ML model. Error: \(error.localizedDescription)")
//        }
//        #endif

        self.tokenizer = BertTokenizer()
    }

    // MARK: - Dense Embeddings

    public func encode(sentence: String) async -> [Float]? {
        // Encode input text as bert tokens
        let inputTokens = tokenizer.buildModelTokens(sentence: sentence)
//        print("Input Token Length: \(inputTokens.count)")
        let (inputIds, attentionMask) = tokenizer.buildModelInputs(from: inputTokens)

        // Send tokens through the MLModel
        let embeddings = generateGteSmallEmbeddings(inputIds: inputIds,
                                                    attentionMask: attentionMask)

        return embeddings
    }

    public func generateGteSmallEmbeddings(inputIds: MLMultiArray, attentionMask: MLMultiArray) -> [Float]? {
        
        //GTE-Small specific
        let tokenTypeValue = 0
        
        //take inputIds and create an array of
        let tokenIdCount = inputIds.shape[1]
//        print("Token Id Count: \(tokenIdCount)")
        
        //make an array of tokenTypeValue of length tokenIdCount
        let tokenTypeIdValues = Array(repeating: tokenTypeValue, count: tokenIdCount.intValue)
//        print("Token Type Id Values: \(tokenTypeIdValues.count)")
        
        let tokenTypeIds = MLMultiArray.from(tokenTypeIdValues, dims: 2)
        
        let inputFeatures = thenlper_gte_smallInput(
            input_ids: inputIds,
            token_type_ids: tokenTypeIds,
            attention_mask: attentionMask
        )
//#if os(macOS)
//        let inputFeatures = thenlper_gte_smallInput(
//            input_ids: inputIds,
//            token_type_ids: tokenTypeIds,
//            attention_mask: attentionMask
//        )
//#else
//        let inputFeatures = thenlper_gte_small_quantized_8_bitInput(
//            input_ids: inputIds,
//            token_type_ids: tokenTypeIds,
//            attention_mask: attentionMask
//        )
//#endif

        let output = try? model.prediction(input: inputFeatures)

        guard let embeddings = output?.embeddings else {
            return nil
        }

        let embeddingsArray: [Float] = (0..<embeddings.count).map { Float(embeddings[$0].floatValue) }
        
        let mag = embeddingsArray.magnitude();
//        print("Magnitude: \(mag)")
        
        let normalized = embeddingsArray.normalized();
        let normalizedMag = normalized?.magnitude();
        
//        print("Normalized: \(normalizedMag ?? -1.0)")
        
        return normalized
        

//        return embeddingsArray
    }
}
