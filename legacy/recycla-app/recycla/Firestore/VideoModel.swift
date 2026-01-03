//
//  VideoModel.swift
//  recycla
//
//  Created by Christopher Minar on 2/13/25.
//

import Foundation
import FirebaseFirestore

struct RecyclableItemModel: Codable {
    /// The Recyclable Item Model is the structure used to pass the item back and forth between firebase
    let material: String?
    let subMaterial: String?
    let category: String?
    let brand: String?

    enum CodingKeys: String, CodingKey {
        case material = "material"
        case subMaterial = "sub_material"
        case category = "category"
        case brand = "brand"
    }
    
    init(
        material: String? = nil,
        subMaterial: String? = nil,
        category: String? = nil,
        brand: String? = nil,
        product: String? = nil
    ) {
        self.material = material
        self.subMaterial = subMaterial
        self.category = category
        self.brand = brand
    }
    
    init(subMaterial: SubMaterial) {
        self.material = Material(subMaterial: subMaterial).snakeCase
        self.subMaterial = subMaterial.snakeCase
        self.category = nil
        self.brand = nil
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.material = try container.decodeIfPresent(String.self, forKey: .material)
        self.subMaterial = try container.decodeIfPresent(String.self, forKey: .subMaterial)
        self.category = try container.decodeIfPresent(String.self, forKey: .category)
        self.brand = try container.decodeIfPresent(String.self, forKey: .brand)
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(self.material, forKey: .material)
        try container.encodeIfPresent(self.subMaterial, forKey: .subMaterial)
        try container.encodeIfPresent(self.category, forKey: .category)
        try container.encodeIfPresent(self.brand, forKey: .brand)
    }
}

extension RecyclableItemModel {
    init(from dictionary: [String: String?]) {
        self.material = (dictionary["material"] ?? "") ?? ""
        self.subMaterial = dictionary["sub_material"] ?? nil
        self.category = dictionary["category"] ?? nil
        self.brand = dictionary["brand"] ?? nil
    }
}

class VideoModelListener: ObservableObject {
    private var listener: ListenerRegistration?
    @Published var videoModel: VideoModel?
    @Published var predictedItem: RecyclableItem? = nil
    @Published var userGroundTruth: RecyclableItem? = nil

    func listenToVideoDocument(id: String, completion: ((VideoModel?) -> Void)? = nil) {
        let db = Firestore.firestore()
        listener = db.collection("videos").document(id).addSnapshotListener { snapshot, error in
            guard let snapshot = snapshot, snapshot.exists,
                  let data = try? snapshot.data(as: VideoModel.self) else {
                completion?(nil)
                return
            }
            self.videoModel = data
            self.predictedItem = RecyclableItem(model: data.classId)
            self.userGroundTruth = data.userGroundTruth != nil ? RecyclableItem(model: data.userGroundTruth!) : nil
            completion?(data)
        }
    }

    func removeListener() {
        listener?.remove()
    }
}

struct VideoModel: Codable {
    let id: String
    let classId: RecyclableItemModel
    let confidence: Double
    let piid: String
    let previewGcsPath: String
    let videoGcsPath: String
    let uploadDate: Date
    let processDate: Date
    let userGroundTruth: RecyclableItemModel?
    let firmwareVersion: String?
    let report: String?
    let groceryItem: String?
    
    enum CodingKeys: String, CodingKey {
        case id = "id"
        case classId = "class_id"
        case confidence = "confidence"
        case piid = "piid"
        case previewGcsPath = "preview_gcs_path"
        case videoGcsPath = "video_gcs_path"
        case uploadDate = "upload_date"
        case processDate = "process_date"
        case userGroundTruth = "user_ground_truth"
        case firmwareVersion = "firmware_version"
        case report = "report"
        case groceryItem = "grocery_item"
    }
    
    init(
        id: String,
        classId: [String: String],
        confidence: Double,
        piid: String,
        previewGcsPath: String,
        videoGcsPath: String,
        uploadDate: Date,
        processDate: Date,
        userGroundTruth: [String: String]? = nil,
        firmwareVersion: String? = nil,
        report: String? = nil,
        groceryItem: String? = nil
    ) {
        self.id = id
        self.classId = RecyclableItemModel(from: classId)
        self.confidence = confidence
        self.piid = piid
        self.previewGcsPath = previewGcsPath
        self.videoGcsPath = videoGcsPath
        self.uploadDate = uploadDate
        self.processDate = processDate
        self.userGroundTruth = userGroundTruth != nil ? RecyclableItemModel(from: userGroundTruth!) : nil
        self.firmwareVersion = firmwareVersion
        self.report = report
        self.groceryItem = groceryItem
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try container.decode(String.self, forKey: .id)
        self.classId = try container.decode(RecyclableItemModel.self, forKey: .classId)
        self.confidence = try container.decode(Double.self, forKey: .confidence)
        self.piid = try container.decode(String.self, forKey: .piid)
        self.previewGcsPath = try container.decode(String.self, forKey: .previewGcsPath)
        self.videoGcsPath = try container.decode(String.self, forKey: .videoGcsPath)
        self.uploadDate = try container.decode(Date.self, forKey: .uploadDate)
        self.processDate = try container.decode(Date.self, forKey: .processDate)
        self.userGroundTruth = try container.decodeIfPresent(RecyclableItemModel.self, forKey: .userGroundTruth)
        self.firmwareVersion = try container.decodeIfPresent(String.self, forKey: .firmwareVersion)
        self.report = try container.decodeIfPresent(String.self, forKey: .report)
        self.groceryItem = try container.decodeIfPresent(String.self, forKey: .groceryItem)
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(self.id, forKey: .id)
        try container.encode(self.classId, forKey: .classId)
        try container.encode(self.confidence, forKey: .confidence)
        try container.encode(self.piid, forKey: .piid)
        try container.encode(self.previewGcsPath, forKey: .previewGcsPath)
        try container.encode(self.videoGcsPath, forKey: .videoGcsPath)
        try container.encode(self.uploadDate, forKey: .uploadDate)
        try container.encode(self.processDate, forKey: .processDate)
        try container.encode(self.userGroundTruth, forKey: .userGroundTruth)
        try container.encode(self.firmwareVersion, forKey: .firmwareVersion)
        try container.encode(self.report, forKey: .report)
        try container.encode(self.groceryItem, forKey: .groceryItem)
    }
}
