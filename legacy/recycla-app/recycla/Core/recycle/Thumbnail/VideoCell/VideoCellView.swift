//
//  VideoCellView.swift
//  recycla
//
//  Created by Christopher Minar on 1/30/25.
//

import SwiftUI
import Foundation
import Combine

/// Determines the `SubMaterial` to use based on the predicted and ground truth items.
func whatAreWeDealingWithHere(predictedItem: RecyclableItem, groundTruthItem: RecyclableItem?) -> SubMaterial? {
    return groundTruthItem?.subMaterial ?? predictedItem.subMaterial
}

struct VideoCellView: View {
    @ObservedObject private var thumbnailManager = ThumbnailManager.shared

    let videoId: String
    @StateObject private var listener = VideoModelListener()
    @State private var selected: Bool = false
    @State private var previewURL: URL? = nil

    init(videoId:String) {
        self.videoId = videoId
        self.selected = false
    }

    private func checkRecycleFilter() -> Bool {
        if listener.videoModel != nil {
            if let ugt = listener.userGroundTruth {
                /// don't allow it to update until the submaterial changes, which triggers the change to the backend
                if ugt.subMaterial != nil {
                    let menuOptions = thumbnailManager.menuOptions
                    if [.glass, .metal, .paper, .plastic, .mixed].contains(ugt.material) {
                        return menuOptions[0].isChecked
                    } else if [.waste, .compost].contains(ugt.material) {
                        return menuOptions[1].isChecked
                    } else if [.miscellaneous].contains(ugt.material) {
                        return menuOptions[2].isChecked
                    }
                }
            }
        }
        return true
    }

    private func checkDateFilter() -> Bool {
        if let video = listener.videoModel {
            let date = video.uploadDate
            let calendar = Calendar.current
            let now = Date()

            for option in thumbnailManager.timeFilterOptions {
                if option.isSelected {
                    switch option.label {
                    case "Day":
                        return calendar.isDate(date, inSameDayAs: now)
                    case "Week":
                        if let weekAgo = calendar.date(byAdding: .day, value: -7, to: now) {
                            return date >= weekAgo
                        }
                    case "Month":
                        if let monthAgo = calendar.date(byAdding: .month, value: -1, to: now) {
                            return date >= monthAgo
                        }
                    case "All":
                        return true
                    default:
                        return true
                    }
                }
            }
        }
        return true
    }

    var body: some View {
        if checkRecycleFilter() && checkDateFilter() {
            VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .top, spacing: 12) {
                    VideoCellImageView(previewURL: previewURL, selected: $selected)

                    if !self.selected {
                        VideoCellInfoView(video: listener.videoModel, predictedItem: $listener.predictedItem, groundTruth: $listener.userGroundTruth, selected: selected)
                    }
                }.onTapGesture {
                    selected.toggle()
                }
                if self.selected {
                    VStack {
                        if let video = listener.videoModel {
                            VideoCellInfoView(video: video, predictedItem: $listener.predictedItem, groundTruth: $listener.userGroundTruth, selected: selected)
                            GroundTruthUpdateView(video: video, prediction: $listener.predictedItem, groundTruth: $listener.userGroundTruth)
                        }
                    }
                }
            }
            .padding(.leading, 4)
            .onAppear {
                listener.listenToVideoDocument(id: videoId) { videoModel in
                    Task {
                        previewURL = try? await ThumbnailManager.shared.getPreviewUrl(for: videoModel?.previewGcsPath)
                    }
                }
            }
            .onDisappear {
                listener.removeListener()
            }
        }
    }
}

struct VideoCellImageView: View {
    let previewURL: URL?
    @Binding var selected: Bool

    var body: some View{
        VStack(alignment: .leading) {
            AsyncImage(url: previewURL ?? URL(string: "")) { image in
                image
                    .resizable()
                    .scaledToFill()
                    .frame(width: self.selected ? 320 : 75, height: self.selected ? 240 : 75)
                    .cornerRadius(10)
            } placeholder: {
                ProgressView()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(Color.clear)
            }
            .frame(width: self.selected ? 320 : 75, height: self.selected ? 240 : 75, alignment: .topLeading)
            .shadow(color: Color.black.opacity(0.3), radius: 4, x: 0, y: 2)

            Spacer()
        }
    }
}

struct VideoCellInfoView: View {
    let video: VideoModel?
    @Binding var predictedItem: RecyclableItem?
    @Binding var groundTruth: RecyclableItem?
    let selected: Bool

    var body: some View {
        if let video = video {
            VStack {
                if let unwrappedPredictedItem = predictedItem {
                    if let title = video.groceryItem {
                        HStack{
                            Text(title.capitalized)
                                .font(.title3)
                                .lineLimit(1)
                                .truncationMode(.tail)
                            Spacer()
                        }
                        PredictionTextSubView(predictedItem: unwrappedPredictedItem, groundTruth: groundTruth)
                    } else {
                        PredictionTextSubView(predictedItem: unwrappedPredictedItem, groundTruth: groundTruth)
                            .minimumScaleFactor(0.5)
                    }

                    let subMaterial = whatAreWeDealingWithHere(predictedItem: unwrappedPredictedItem, groundTruthItem: groundTruth)
                    HStack {
                        IsRecyclableSubView(subMaterial: subMaterial, longString: selected)
                        Spacer()
                        TimeSinceView(uploadDate: video.uploadDate)
                            .padding(.trailing, 2)
                    }
                    .padding(.top, -6)
                } else {
                    TimeSinceView(uploadDate: video.uploadDate)
                }
            }
            .font(.callout)
            .foregroundColor(.secondary)
        }
    }
}

struct GroundTruthUpdateView: View {
    let video: VideoModel
    @Binding var prediction: RecyclableItem?
    @Binding var groundTruth: RecyclableItem?
    @State private var selectedMaterial: Material?
    @State private var selectedSubMaterial: SubMaterial?
    @State private var selectedWhat: String?
    @State private var selectedBrand: String?
    @State private var videoReport: String?

    init(video: VideoModel, prediction: Binding<RecyclableItem?>, groundTruth: Binding<RecyclableItem?>) {
        self.video = video
        self._prediction = prediction
        self._groundTruth = groundTruth
        self._videoReport = State(initialValue: video.report) // Initialize here
    }

    var body: some View {
        VStack() {
            MaterialSelectionView(
                prediction: prediction ?? RecyclableItem(subMaterial: .miscellaneous), // default to misc
                groundTruth: $groundTruth,
                selectedMaterial: $selectedMaterial,
                selectedSubMaterial: $selectedSubMaterial,
                selectedCategory: $selectedWhat,
                selectedBrand: $selectedBrand,
                videoId: video.id,
                videoReport: $videoReport
            )
        }
        .foregroundColor(.secondary)
    }
}

#Preview {
    VStack{
        VideoCellView(videoId: "PTH1o6XcmnMZmBWHpt1a")
    }
}
