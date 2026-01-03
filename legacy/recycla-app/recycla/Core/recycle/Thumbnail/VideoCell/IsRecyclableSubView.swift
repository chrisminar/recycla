//
//  IsRecyclableSubView.swift
//  recycla
//
//  Created by Christopher Minar on 4/2/25.
//

import SwiftUI

struct RecyclabilityIconView: View {
    let material: SubMaterial?

    var body: some View {
        guard let isRecyclableData = material?.isRecyclable else {
            return AnyView(
                Image(systemName: "questionmark")
                    .resizable()
                    .frame(width: 32, height: 32)
            )
        }
        let (recycleGroup, ambiguous, _) = isRecyclableData
        let backgroundImageMap = [1: "recycle-sign_128", 2: "yellow_recycle", 3: "orange_recycle", 4: "trash"]

        // Determine the image based on the recycleGroup
        let backgroundImage: Image
        if let key = backgroundImageMap[recycleGroup] {
            if recycleGroup == 1 || recycleGroup == 2 || recycleGroup == 3 {
                backgroundImage = Image(key) // Use the key as the image name
            } else if recycleGroup == 4 {
                backgroundImage = Image(systemName: key) // Use the key as a system image name
            } else {
                backgroundImage = Image(systemName: "trash.slash") // Default to trash.slash
            }
        } else {
            backgroundImage = Image(systemName: "trash.slash") // Default to trash.slash
        }

        let foregroundImage = Image(systemName: "questionmark")
        let fgIm = ambiguous ? foregroundImage : nil

        if let fgIm = fgIm {
            return AnyView(
                backgroundImage
                    .resizable()
                    .frame(width: 32, height: 32)
                    .overlay {
                        fgIm
                            .resizable()
                            .scaledToFit()
                            .frame(width: 28, height: 28)
                            .shadow(color: Color.black, radius: 1, x: 1, y: 1) // Add a shadow
                    }
            )
        } else {
            return AnyView(
                backgroundImage
                    .resizable()
                    .frame(width: 32, height: 32)
            )
        }
    }
}

struct IsRecyclableSubView: View {
    let subMaterial: SubMaterial?
    let longString: Bool

    func recyclableText(material: SubMaterial?) -> String {
        guard let isRecyclableData = material?.isRecyclable else {
            return "?"
        }
        let (recycleGroup, _, recyclabilityString) = isRecyclableData

        if !longString {
            switch recycleGroup {
            case 1:
                return "recyclable"
            case 2:
                return "sometimes recyclable"
            case 3:
                return "rarely recyclable"
            case 4:
                return "not recyclable"
            default:
                return "not disposable"
            }
        } else {
            return recyclabilityString
        }
    }

    var body: some View {
        HStack {
            RecyclabilityIconView(material: subMaterial)
            Text(recyclableText(material: subMaterial))
                .foregroundColor(Color.black)
            Spacer()
        }
    }
}

#Preview {
    let materials = [
        SubMaterial.glassBottle, // recyclable
        SubMaterial.mixedTetraPak, // sometimes
        SubMaterial.plastic, // sometimes ambiguous
        SubMaterial.mixedSpiralWound, // rarely
        SubMaterial.waste, // never
        SubMaterial.miscellaneousTest // not disposable
    ]
    List {
        ForEach(materials, id: \.self) { material in
            IsRecyclableSubView(subMaterial: material, longString: true)
        }
    }
}
