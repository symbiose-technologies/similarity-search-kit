//
// thenlper_gte_small.swift
//
// This file was automatically generated and should not be edited.
//

import CoreML


/// Model Prediction Input Type
@available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, visionOS 1.0, *)
class thenlper_gte_smallInput : MLFeatureProvider {

    /// input_ids as 1 by 512 matrix of 32-bit integers
    var input_ids: MLMultiArray

    /// token_type_ids as 1 by 512 matrix of 32-bit integers
    var token_type_ids: MLMultiArray

    /// attention_mask as 1 by 512 matrix of 32-bit integers
    var attention_mask: MLMultiArray

    var featureNames: Set<String> { ["input_ids", "token_type_ids", "attention_mask"] }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "input_ids" {
            return MLFeatureValue(multiArray: input_ids)
        }
        if featureName == "token_type_ids" {
            return MLFeatureValue(multiArray: token_type_ids)
        }
        if featureName == "attention_mask" {
            return MLFeatureValue(multiArray: attention_mask)
        }
        return nil
    }

    init(input_ids: MLMultiArray, token_type_ids: MLMultiArray, attention_mask: MLMultiArray) {
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
    }

    convenience init(input_ids: MLShapedArray<Int32>, token_type_ids: MLShapedArray<Int32>, attention_mask: MLShapedArray<Int32>) {
        self.init(input_ids: MLMultiArray(input_ids), token_type_ids: MLMultiArray(token_type_ids), attention_mask: MLMultiArray(attention_mask))
    }

}


/// Model Prediction Output Type
@available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, visionOS 1.0, *)
class thenlper_gte_smallOutput : MLFeatureProvider {

    /// Source provided by CoreML
    private let provider : MLFeatureProvider

    /// embeddings as multidimensional array of floats
    var embeddings: MLMultiArray {
        provider.featureValue(for: "embeddings")!.multiArrayValue!
    }

    /// embeddings as multidimensional array of floats
    var embeddingsShapedArray: MLShapedArray<Float> {
        MLShapedArray<Float>(embeddings)
    }

    var featureNames: Set<String> {
        provider.featureNames
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        provider.featureValue(for: featureName)
    }

    init(embeddings: MLMultiArray) {
        self.provider = try! MLDictionaryFeatureProvider(dictionary: ["embeddings" : MLFeatureValue(multiArray: embeddings)])
    }

    init(features: MLFeatureProvider) {
        self.provider = features
    }
}


/// Class for model loading and prediction
@available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, visionOS 1.0, *)
class thenlper_gte_small {
    let model: MLModel

    /// URL of model assuming it was installed in the same bundle as this class
    class var urlOfModelInThisBundle : URL {
        let bundle = Bundle(for: self)
        return bundle.url(forResource: "thenlper-gte-small", withExtension:"mlmodelc")!
    }

    /**
        Construct thenlper_gte_small instance with an existing MLModel object.

        Usually the application does not use this initializer unless it makes a subclass of thenlper_gte_small.
        Such application may want to use `MLModel(contentsOfURL:configuration:)` and `thenlper_gte_small.urlOfModelInThisBundle` to create a MLModel object to pass-in.

        - parameters:
          - model: MLModel object
    */
    init(model: MLModel) {
        self.model = model
    }

    /**
        Construct a model with configuration

        - parameters:
           - configuration: the desired model configuration

        - throws: an NSError object that describes the problem
    */
    convenience init(configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        try self.init(contentsOf: type(of:self).urlOfModelInThisBundle, configuration: configuration)
    }

    /**
        Construct thenlper_gte_small instance with explicit path to mlmodelc file
        - parameters:
           - modelURL: the file url of the model

        - throws: an NSError object that describes the problem
    */
    convenience init(contentsOf modelURL: URL) throws {
        try self.init(model: MLModel(contentsOf: modelURL))
    }

    /**
        Construct a model with URL of the .mlmodelc directory and configuration

        - parameters:
           - modelURL: the file url of the model
           - configuration: the desired model configuration

        - throws: an NSError object that describes the problem
    */
    convenience init(contentsOf modelURL: URL, configuration: MLModelConfiguration) throws {
        try self.init(model: MLModel(contentsOf: modelURL, configuration: configuration))
    }

    /**
        Construct thenlper_gte_small instance asynchronously with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - configuration: the desired model configuration
          - handler: the completion handler to be called when the model loading completes successfully or unsuccessfully
    */
    class func load(configuration: MLModelConfiguration = MLModelConfiguration(), completionHandler handler: @escaping (Swift.Result<thenlper_gte_small, Error>) -> Void) {
        load(contentsOf: self.urlOfModelInThisBundle, configuration: configuration, completionHandler: handler)
    }

    /**
        Construct thenlper_gte_small instance asynchronously with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - configuration: the desired model configuration
    */
    class func load(configuration: MLModelConfiguration = MLModelConfiguration()) async throws -> thenlper_gte_small {
        try await load(contentsOf: self.urlOfModelInThisBundle, configuration: configuration)
    }

    /**
        Construct thenlper_gte_small instance asynchronously with URL of the .mlmodelc directory with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - modelURL: the URL to the model
          - configuration: the desired model configuration
          - handler: the completion handler to be called when the model loading completes successfully or unsuccessfully
    */
    class func load(contentsOf modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration(), completionHandler handler: @escaping (Swift.Result<thenlper_gte_small, Error>) -> Void) {
        MLModel.load(contentsOf: modelURL, configuration: configuration) { result in
            switch result {
            case .failure(let error):
                handler(.failure(error))
            case .success(let model):
                handler(.success(thenlper_gte_small(model: model)))
            }
        }
    }

    /**
        Construct thenlper_gte_small instance asynchronously with URL of the .mlmodelc directory with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - modelURL: the URL to the model
          - configuration: the desired model configuration
    */
    class func load(contentsOf modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration()) async throws -> thenlper_gte_small {
        let model = try await MLModel.load(contentsOf: modelURL, configuration: configuration)
        return thenlper_gte_small(model: model)
    }

    /**
        Make a prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - input: the input to the prediction as thenlper_gte_smallInput

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as thenlper_gte_smallOutput
    */
    func prediction(input: thenlper_gte_smallInput) throws -> thenlper_gte_smallOutput {
        try prediction(input: input, options: MLPredictionOptions())
    }

    /**
        Make a prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - input: the input to the prediction as thenlper_gte_smallInput
           - options: prediction options

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as thenlper_gte_smallOutput
    */
    func prediction(input: thenlper_gte_smallInput, options: MLPredictionOptions) throws -> thenlper_gte_smallOutput {
        let outFeatures = try model.prediction(from: input, options: options)
        return thenlper_gte_smallOutput(features: outFeatures)
    }

    /**
        Make an asynchronous prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - input: the input to the prediction as thenlper_gte_smallInput
           - options: prediction options

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as thenlper_gte_smallOutput
    */
    @available(macOS 14.0, iOS 17.0, tvOS 17.0, watchOS 10.0, visionOS 1.0, *)
    func prediction(input: thenlper_gte_smallInput, options: MLPredictionOptions = MLPredictionOptions()) async throws -> thenlper_gte_smallOutput {
        let outFeatures = try await model.prediction(from: input, options: options)
        return thenlper_gte_smallOutput(features: outFeatures)
    }

    /**
        Make a prediction using the convenience interface

        It uses the default function if the model has multiple functions.

        - parameters:
            - input_ids: 1 by 512 matrix of 32-bit integers
            - token_type_ids: 1 by 512 matrix of 32-bit integers
            - attention_mask: 1 by 512 matrix of 32-bit integers

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as thenlper_gte_smallOutput
    */
    func prediction(input_ids: MLMultiArray, token_type_ids: MLMultiArray, attention_mask: MLMultiArray) throws -> thenlper_gte_smallOutput {
        let input_ = thenlper_gte_smallInput(input_ids: input_ids, token_type_ids: token_type_ids, attention_mask: attention_mask)
        return try prediction(input: input_)
    }

    /**
        Make a prediction using the convenience interface

        It uses the default function if the model has multiple functions.

        - parameters:
            - input_ids: 1 by 512 matrix of 32-bit integers
            - token_type_ids: 1 by 512 matrix of 32-bit integers
            - attention_mask: 1 by 512 matrix of 32-bit integers

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as thenlper_gte_smallOutput
    */

    func prediction(input_ids: MLShapedArray<Int32>, token_type_ids: MLShapedArray<Int32>, attention_mask: MLShapedArray<Int32>) throws -> thenlper_gte_smallOutput {
        let input_ = thenlper_gte_smallInput(input_ids: input_ids, token_type_ids: token_type_ids, attention_mask: attention_mask)
        return try prediction(input: input_)
    }

    /**
        Make a batch prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - inputs: the inputs to the prediction as [thenlper_gte_smallInput]
           - options: prediction options

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as [thenlper_gte_smallOutput]
    */
    func predictions(inputs: [thenlper_gte_smallInput], options: MLPredictionOptions = MLPredictionOptions()) throws -> [thenlper_gte_smallOutput] {
        let batchIn = MLArrayBatchProvider(array: inputs)
        let batchOut = try model.predictions(from: batchIn, options: options)
        var results : [thenlper_gte_smallOutput] = []
        results.reserveCapacity(inputs.count)
        for i in 0..<batchOut.count {
            let outProvider = batchOut.features(at: i)
            let result =  thenlper_gte_smallOutput(features: outProvider)
            results.append(result)
        }
        return results
    }
}
