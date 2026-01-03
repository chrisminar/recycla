//
//  PiModel.swift
//  recycla
//
//  Created by Christopher Minar on 2/13/25.
//

import Foundation

struct PiModel: Identifiable, Codable {
    let id: String
    let dateAdded: Date?
    let videos: [String]?
    let users: [String]?

    let numItems: Int?
    let materialPctCorrect: Double?
    let subMaterialPctCorrect: Double?
    let materialNumClass: [String: Int]?
    let subMaterialNumClass: [String: Int]?
    let materialPctClassCorrect: [String: Double]?
    let subMaterialPctClassCorrect: [String: Double]?
    
    init (
        id: String,
        dateAdded: Date? = nil,
        email: String? = nil,
        videos: [String]? = nil,
        users: [String]? = nil,
        numItems: Int? = nil,
        materialPctCorrect: Double? = nil,
        subMaterialPctCorrect: Double? = nil,
        materialNumClass: [String: Int]? = nil,
        subMaterialNumClass: [String: Int]? = nil,
        materialPctClassCorrect: [String: Double]? = nil,
        subMaterialPctClassCorrect: [String: Double]? = nil,
    ) {
        self.id = id
        self.dateAdded = dateAdded
        self.videos = videos
        self.users = users
        self.numItems = numItems
        self.materialPctCorrect = materialPctCorrect
        self.subMaterialPctCorrect = subMaterialPctCorrect
        self.materialNumClass = materialNumClass
        self.subMaterialNumClass = subMaterialNumClass
        self.materialPctClassCorrect = materialPctClassCorrect
        self.subMaterialPctClassCorrect = subMaterialPctClassCorrect
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try container.decode(String.self, forKey: .id)
        self.dateAdded = try container.decodeIfPresent(Date.self, forKey: .dateAdded)
        self.videos = try container.decodeIfPresent([String].self, forKey: .videos)
        self.users = try container.decodeIfPresent([String].self, forKey: .users)
        self.numItems = try container.decodeIfPresent(Int.self, forKey: .numItems)
        self.materialPctCorrect = try container.decodeIfPresent(Double.self, forKey: .materialPctCorrect)
        self.subMaterialPctCorrect = try container.decodeIfPresent(Double.self, forKey: .subMaterialPctCorrect)
        self.materialNumClass = try container.decodeIfPresent([String: Int].self, forKey: .materialNumClass)
        self.subMaterialNumClass = try container.decodeIfPresent([String: Int].self, forKey: .subMaterialNumClass)
        self.materialPctClassCorrect = try container.decodeIfPresent([String: Double].self, forKey: .materialPctClassCorrect)
        self.subMaterialPctClassCorrect = try container.decodeIfPresent([String: Double].self, forKey: .subMaterialPctClassCorrect)
    }
}

/// I wanted a version of pimodel without the id field and couldn't figure out how to do it elegantly so I 
/// spaghetti coded it
struct PiModelNoID: Codable {
    let numItems: Int?
    let materialPctCorrect: Double?
    let subMaterialPctCorrect: Double?
    let materialNumClass: [String: Int]?
    let subMaterialNumClass: [String: Int]?
    let materialPctClassCorrect: [String: Double]?
    let subMaterialPctClassCorrect: [String: Double]?
    
    init (
        numItems: Int? = nil,
        materialPctCorrect: Double? = nil,
        subMaterialPctCorrect: Double? = nil,
        materialNumClass: [String: Int]? = nil,
        subMaterialNumClass: [String: Int]? = nil,
        materialPctClassCorrect: [String: Double]? = nil,
        subMaterialPctClassCorrect: [String: Double]? = nil,
    ) {
        self.numItems = numItems
        self.materialPctCorrect = materialPctCorrect
        self.subMaterialPctCorrect = subMaterialPctCorrect
        self.materialNumClass = materialNumClass
        self.subMaterialNumClass = subMaterialNumClass
        self.materialPctClassCorrect = materialPctClassCorrect
        self.subMaterialPctClassCorrect = subMaterialPctClassCorrect
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.numItems = try container.decodeIfPresent(Int.self, forKey: .numItems)
        self.materialPctCorrect = try container.decodeIfPresent(Double.self, forKey: .materialPctCorrect)
        self.subMaterialPctCorrect = try container.decodeIfPresent(Double.self, forKey: .subMaterialPctCorrect)
        self.materialNumClass = try container.decodeIfPresent([String: Int].self, forKey: .materialNumClass)
        self.subMaterialNumClass = try container.decodeIfPresent([String: Int].self, forKey: .subMaterialNumClass)
        self.materialPctClassCorrect = try container.decodeIfPresent([String: Double].self, forKey: .materialPctClassCorrect)
        self.subMaterialPctClassCorrect = try container.decodeIfPresent([String: Double].self, forKey: .subMaterialPctClassCorrect)
    }
}
