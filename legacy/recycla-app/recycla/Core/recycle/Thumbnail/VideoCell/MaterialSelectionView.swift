//
//  MaterialSelectionView.swift
//  recycla
//
//  Created by Christopher Minar on 4/2/25.
//

import SwiftUI

struct MaterialSelectionView: View {
    let prediction: RecyclableItem
    @Binding var groundTruth: RecyclableItem?
    @Binding var selectedMaterial: Material?
    @Binding var selectedSubMaterial: SubMaterial?
    @Binding var selectedCategory: String?
    @Binding var selectedBrand: String?
    let videoId: String
    @Binding var videoReport: String?

    var body: some View {
        VStack {
            Text("Update ground truth")
                .font(.headline)
                .padding(.horizontal)

            Divider()
                .padding(.horizontal)
                .padding(.top, -6)
            
            VStack{
                HStack {
                    // Material selection menu
                    createMenu(
                        label: selectedMaterial?.rawValue ?? "Material",
                        items: Material.allCases,
                        isDisabled: false,
                        backgroundColor: selectedMaterial == nil ? Color.blue.opacity(0.5) : Color.blue,
                        action: { material in
                            handleMaterialSelection(material)
                        }
                    )
                    
                    // SubMaterial selection menu
                    createMenu(
                        label: selectedSubMaterial?.justSub ?? "SubMaterial",
                        items: selectedMaterial?.subMaterials ?? [],
                        isDisabled: selectedMaterial == nil,
                        backgroundColor: selectedMaterial == nil ? Color.gray.opacity(0.5) : selectedSubMaterial == nil ? Color.blue.opacity(0.5) : Color.blue,
                        action: { subMaterial in
                            handleSubMaterialSelection(subMaterial)
                        }
                    )
                }
                .padding(.horizontal)
                .padding(.top, -4)

                // Add TextInputMenus for whatIs, Brand, and Product
                // only display if submaterial is good
                let sm = whatAreWeDealingWithHere(predictedItem: prediction, groundTruthItem: groundTruth)
                if sm?.brandIsInteresting ?? false {
                    HStack(spacing: 10) {
                        TextInputMenu(
                            selectedText: $selectedCategory,
                            label: selectedCategory ?? "Category",
                            placeholder: "Enter description: e.g. Soda",
                            action: { text in
                                handleTextSelection(text, for: "category")
                            }
                        )
                        TextInputMenu(
                            selectedText: $selectedBrand,
                            label: selectedBrand ?? "Brand",
                            placeholder: "Enter brand: e.g. Coca-Cola",
                            action: { text in
                                handleTextSelection(text, for: "brand")
                            }
                        )
                    }
                    .padding(.horizontal)
                }
                
                // Add report button
                TextInputMenu(
                    selectedText: $videoReport,
                    label: "Report",
                    placeholder: "What went wrong?",
                    action: { text in
                        handleReportSelection(text)
                    }
                )
                .backgroundColor(selected: Color.red, notSelected: Color.red.opacity(0.5))
                .systemImage("exclamationmark.triangle.fill")
                .defaultOptions(["Blurry Preview", "False Positive", "Bad Preview Image", "Mixed Material", "Item Dirty"])
                .padding(.horizontal)
            }
        }
        .onAppear {
            /// Initialize selectedMaterial if groundTruth is not nil
            if let groundTruth = groundTruth {
                selectedMaterial = groundTruth.material
                selectedSubMaterial = groundTruth.subMaterial
                selectedCategory = groundTruth.category
                selectedBrand = groundTruth.brand
            } else {
                /// otherwise, make a new ground truth holder
                groundTruth = RecyclableItem(material: selectedMaterial, subMaterial: selectedSubMaterial, category: selectedCategory, brand: selectedBrand, product: nil)
            }
        }
    }

    // Helper function to create a menu
    private func createMenu<T: Hashable>(
        label: String,
        items: [T],
        isDisabled: Bool,
        backgroundColor: Color,
        action: @escaping (T) -> Void
    ) -> some View {
        Menu {
            if items.isEmpty {
                Text("No Items Available")
                    .foregroundColor(.gray)
            } else {
                ForEach(items, id: \.self) { item in
                    Button(action: {
                        action(item)
                    }) {
                        if let subMaterial = item as? SubMaterial {
                            Text(subMaterial.pretty)
                        } else {
                            Text("\(item)") // Default for other types
                        }
                    }
                }
            }
        } label: {
            Label(label, systemImage: "arrowtriangle.down.fill")
                .font(.headline)
                .foregroundColor(isDisabled ? .gray : .white)
                .frame(height: 30)
                .frame(maxWidth: .infinity)
                .background(backgroundColor)
                .cornerRadius(10)
        }
        .disabled(isDisabled)
    }

    // Handle material selection
    private func handleMaterialSelection(_ material: Material) {
        /// If ground truth is not nil, overwrite it with selected values
        if groundTruth != nil {
            self.groundTruth?.material = material
            self.groundTruth?.subMaterial = nil
        } else {
            print("Error, ground truth should not be nil here.")
        }

        selectedMaterial = material
        selectedSubMaterial = nil // Reset sub-material when material changes
        /// Don't do anything with the other selecteds
    }

    /// Handle sub-material selection
    private func handleSubMaterialSelection(_ subMaterial: SubMaterial) {
        if groundTruth != nil {
            self.groundTruth?.subMaterial = subMaterial
        } else {
            print("Unexpected error: groundTruth is nil")
        }

        selectedSubMaterial = subMaterial

        /// Upload new ground truth
        updateGroundTruth()
    }

    private func handleTextSelection(_ text: String, for type: String) {
        var newVal: String = ""
        switch type {
        case "category":
            newVal = text
            selectedCategory = text
            groundTruth?.category = text
        case "brand":
            newVal = text
            selectedBrand = text
            groundTruth?.brand = text
        default:
            break
        }
        updateProductGroundTruth(newVal: newVal, key: type)
    }
    
    private func handleReportSelection(_ text:String) {
        self.videoReport = text
        Task {
            do {
                try await ThumbnailManager.shared.updateReportError(videoId: videoId, report: text)
            } catch {
                print("Failed to update report: \(error.localizedDescription)")
            }
        }
    }

    // Update ground truth
    private func updateGroundTruth() {
        Task {
            do {
                if let groundTruth = groundTruth {
                    try await ThumbnailManager.shared.updateGroundTruth(videoId: videoId, newGroundTruth: groundTruth)
                }
            } catch {
                print("Failed to update ground truth: \(error.localizedDescription)")
            }
        }
    }

    // update product ground truth
    private func updateProductGroundTruth(newVal: String?, key: String) {
        Task {
            do {
                if let newVal = newVal {
                    try await ThumbnailManager.shared.updateProductGroundTruth(videoId: videoId, newValue: newVal, for: key)
                }
            } catch {
                print("Failed to update \(key) ground truth: \(error.localizedDescription)")
            }
        }
    }
}

#Preview {
    @State @Previewable var gt: RecyclableItem? = nil//RecyclableItem(subMaterial: SubMaterial.glassBottle)
    @State @Previewable var mat: Material? = nil
    @State @Previewable var sm: SubMaterial? = nil
    @State @Previewable var category: String? = nil
    @State @Previewable var brand: String? = nil
    @State @Previewable var report: String? = nil
    let pred = RecyclableItem(subMaterial: .glassBottle)
    MaterialSelectionView(prediction: pred, groundTruth: $gt, selectedMaterial: $mat, selectedSubMaterial: $sm, selectedCategory: $category, selectedBrand: $brand, videoId: "", videoReport: $report)
}
