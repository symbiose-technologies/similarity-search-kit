//
//  VectorStoreProtocol.swift
//
//
//  Created by Zach Nagengast on 4/26/23.
//

import Foundation

public protocol VectorStoreProtocol {
    func saveIndex(items: [IndexItem], to url: URL, as name: String) throws -> URL
    func loadIndex(from url: URL) throws -> [IndexItem]
    func listIndexes(at url: URL) -> [URL]
    
    
    func acceptAddedItem(item: IndexItem) -> Void
    var jitItems: [IndexItem] { get }
    var isJitProvider: Bool { get }

    //    func acceptUpdatedItem(item: IndexItem) -> Void
    //    var retrievesItemsJustInTime: Bool { get }
}

public extension VectorStoreProtocol {

    var jitItems: [IndexItem] { [] }
    var isJitProvider: Bool { false }
    func acceptAddedItem(item: IndexItem) -> Void { }
    
    /*
     REMOVE, REMOVE ALL, and UPDATE via SimilarityIndex are NOT passed through
     */
    
//    var shouldReceiveUpdatesLive: Bool { false }
//    var retrievesItemsJustInTime: Bool { false }
//
//    func acceptUpdatedItem(item: IndexItem) -> Void { }
    
}
