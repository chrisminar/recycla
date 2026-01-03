//
//  ThumbnailManager.swift
//  recycla
//
//  Created by Christopher Minar on 2/6/25.
//


import Foundation
import FirebaseFirestore
import FirebaseStorage
import SwiftUI

final class ThumbnailManager: ObservableObject {
    
    static let shared = ThumbnailManager()
    private init() {
        loadFilters()
    }
    
    @Published var menuOptions = [
        MenuOption(label: "Recycle", isChecked: true),
        MenuOption(label: "Waste/Compost", isChecked: false),
        MenuOption(label: "Misc", isChecked: false)
    ] {
        didSet {
            saveFilters()
        }
    }
    
    @Published var timeFilterOptions = [
        TimeFilterOption(label: "Day", isSelected: false),
        TimeFilterOption(label: "Week", isSelected: true),
        TimeFilterOption(label: "Month", isSelected: false),
        TimeFilterOption(label: "All", isSelected: false)
    ] {
        didSet {
            saveFilters()
        }
    }

    private func saveFilters() {
        let menuOptionsData = menuOptions.map { ["label": $0.label, "isChecked": $0.isChecked] }
        let timeFilterOptionsData = timeFilterOptions.map { ["label": $0.label, "isSelected": $0.isSelected] }
        
        UserDefaults.standard.set(menuOptionsData, forKey: "menuOptions")
        UserDefaults.standard.set(timeFilterOptionsData, forKey: "timeFilterOptions")
    }

    private func loadFilters() {
        if let menuOptionsData = UserDefaults.standard.array(forKey: "menuOptions") as? [[String: Any]] {
            menuOptions = menuOptionsData.compactMap { dict in
                guard let label = dict["label"] as? String, let isChecked = dict["isChecked"] as? Bool else { return nil }
                return MenuOption(label: label, isChecked: isChecked)
            }
        }
        
        if let timeFilterOptionsData = UserDefaults.standard.array(forKey: "timeFilterOptions") as? [[String: Any]] {
            timeFilterOptions = timeFilterOptionsData.compactMap { dict in
                guard let label = dict["label"] as? String, let isSelected = dict["isSelected"] as? Bool else { return nil }
                return TimeFilterOption(label: label, isSelected: isSelected)
            }
        }
    }

    private let videoCollection = Firestore.firestore().collection("videos")
    
    private func videoDocument(videoId: String) -> DocumentReference {
        videoCollection.document(videoId)
    }
    
    func getVideo(videoId: String) async throws -> VideoModel {
        let doc = try await videoDocument(videoId: videoId).getDocument()
        
        guard let data = doc.data() else {
            throw NSError(domain: "ThumbnailManager", code: 404, userInfo: [NSLocalizedDescriptionKey: "Document not found"])
        }
        
        // Use Firestore's built-in decoder
        let videoModel = try Firestore.Decoder().decode(VideoModel.self, from: data)
        
        return videoModel
    }
    
    func updateGroundTruth(videoId: String, newGroundTruth: RecyclableItem) async throws {
        try await videoDocument(videoId: videoId).updateData([
            "\(VideoModel.CodingKeys.userGroundTruth.rawValue).material": newGroundTruth.material?.snakeCase,
            "\(VideoModel.CodingKeys.userGroundTruth.rawValue).sub_material": newGroundTruth.subMaterial?.snakeCase,
        ])
    }
    
    func updateReportError(videoId: String, report: String) async throws {
        try await videoDocument(videoId: videoId).updateData([
            "\(VideoModel.CodingKeys.report.rawValue)": report
        ])
    }

    func updateProductGroundTruth(videoId: String, newValue: String, for field: String) async throws {
        try await videoDocument(videoId: videoId).updateData([
        "\(VideoModel.CodingKeys.userGroundTruth.rawValue).\(field)": newValue
    ])
    }
        
    func getPreviewUrl(for gcsUrl: String?) async throws -> URL? {
        let storage = Storage.storage()

        if let gcsurl = gcsUrl {
            let gsReference = storage.reference(withPath: gcsurl)
            
            return try await withCheckedThrowingContinuation { continuation in
                gsReference.downloadURL { url, error in
                    if let error = error {
                        continuation.resume(throwing: error)
                    } else if let url = url {
                        continuation.resume(returning: url)
                    } else {
                        continuation.resume(throwing: URLError(.badURL))
                    }
                }
            }
        } else {
            return nil
        }
    }

    func selectTimeFilter(option: TimeFilterOption) {
        for index in timeFilterOptions.indices {
            timeFilterOptions[index].isSelected = (timeFilterOptions[index].id == option.id)
        }
    }
}

struct MenuOption: Identifiable {
    let id = UUID()
    let label: String
    var isChecked: Bool
}

struct TimeFilterOption: Identifiable {
    let id = UUID()
    let label: String
    var isSelected: Bool
}


