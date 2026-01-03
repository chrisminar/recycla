//
//  PredictionTextSubView.swift
//  recycla
//
//  Created by Christopher Minar on 4/2/25.
//

import SwiftUI

struct PredictionTextSubView: View {
    let predictedItem: RecyclableItem
    let groundTruth: RecyclableItem?
    
    var color: Color {
        let materialCorrect = (groundTruth == nil) || (groundTruth?.material == nil) || (groundTruth?.material == predictedItem.material)
        let subMaterialCorrect = (groundTruth == nil) || (groundTruth?.subMaterial == nil) || (groundTruth?.subMaterial?.nearestLabel == predictedItem.subMaterial)
        if materialCorrect && subMaterialCorrect {
            return Color.green
        } else if materialCorrect && !subMaterialCorrect {
            return Color.yellow
        } else if !materialCorrect && subMaterialCorrect {
            return Color.orange
        } else { // both wrong
            return Color.red
        }
    }
    
    var text: String {
        var temp: String
        if (groundTruth == nil) || (groundTruth?.material == nil) || (groundTruth?.subMaterial == nil) {
            if predictedItem.subMaterial != nil {
                temp = predictedItem.subMaterial?.pretty ?? ""
            } else {
                temp = predictedItem.material?.rawValue ?? ""
            }
        } else { // gt not nil
            if groundTruth?.subMaterial != nil { // gt sub not nil
                temp = groundTruth?.subMaterial?.nearestLabel.pretty ?? ""
            } else {
                temp = groundTruth?.material?.rawValue ?? ""
            }
        }
        return temp
    }
    
    var text2: String {
        if color == Color.green {
            return ""
        } else {
            let temp = predictedItem.subMaterial?.pretty ?? ""
            return "(\(temp))"
        }
    }

    var body: some View {
        HStack {
            Text(text) // Predicted item should never have null sub
                .foregroundColor(color)
                .font(.headline)
            Text(text2)
                .foregroundColor(.primary)
                .font(.footnote)
            Spacer()
        }
    }
}

#Preview {
    let p: [(prediction: SubMaterial?, groundTruth: RecyclableItem?)] = [
        (SubMaterial.metalAluminumCan, nil),
        (SubMaterial.metalAluminumCan, RecyclableItem(subMaterial: .metalAluminumCan)),
        //(SubMaterial.metalAluminumCan, RecyclableItem(model: RecyclableItemModel(material: "metal", subMaterial: nil, category: nil, brand: nil, product: nil))),
        (SubMaterial.metalAluminumCan, RecyclableItem(subMaterial: .metalSteelCan)),
        (SubMaterial.metalAluminumCan, RecyclableItem(subMaterial: .glassBottle)),
        (SubMaterial.metal, RecyclableItem(subMaterial: .metal)),
        (SubMaterial.metal, RecyclableItem(subMaterial: .metalAluminumCan)),
        (SubMaterial.metal, RecyclableItem(subMaterial: .paperMail)),
        (SubMaterial.metal, RecyclableItem(subMaterial: .glass)),
        (SubMaterial.metalAluminumCan, RecyclableItem(subMaterial: .metal))
    ]
    
    VStack {
        ForEach(p.indices, id: \.self) { index in
            let prediction = p[index].prediction
            let groundTruth = p[index].groundTruth
            
            VStack(alignment: .leading) {
                Text("\(prediction?.snakeCase ?? "None")      \(groundTruth?.subMaterial?.snakeCase ?? "None")")
                PredictionTextSubView(
                    predictedItem: RecyclableItem(subMaterial: prediction!),
                    groundTruth: groundTruth
                )
                Divider()
            }
        }
    }
}
