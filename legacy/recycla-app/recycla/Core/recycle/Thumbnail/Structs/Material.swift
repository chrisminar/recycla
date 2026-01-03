//
//  Material.swift
//  recycla
//
//  Created by Christopher Minar on 4/21/25.
//
import Foundation

enum Material: String, CaseIterable {
    case glass = "Glass"
    case metal = "Metal"
    case paper = "Paper"
    case plastic = "Plastic"
    case mixed = "Mixed"
    case waste = "Waste"
    case compost = "Compost"
    case miscellaneous = "Miscellaneous"

    /// Return snake case version of enum
    var snakeCase: String {
        switch self {
        case .glass: return "glass"
        case .metal: return "metal"
        case .paper: return "paper"
        case .plastic: return "plastic"
        case .mixed: return "mixed"
        case .waste: return "waste"
        case .compost: return "compost"
        case .miscellaneous: return "miscellaneous"
        }
    }

    /// Initialize from submaterial
    init(subMaterial: SubMaterial) {
        switch subMaterial {
        case .glass, .glassBottle, .glassContainer, .glassWineBottle:
            self = .glass
        case .metal, .metalAluminumAerosol, .metalAluminumCan, .metalAluminumFoil, .metalSteelAerosol, .metalSteelCan:
            self = .metal
        case .paper, .paperCorrugatedCardboard, .paperEggCarton, .paperMail, .paperPaperbackBook, .paperPaperboard, .paperPaperBag, .paperPaperCup, .paperPaperTowel, .paperPizzaBox, .paperShreddedPaper:
            self = .paper
        case .plastic, .plasticBag, .plasticBottle, .plasticPackaging, .plasticTub, .plasticUtensil, .plasticFilm:
            self = .plastic
        case .mixedSpiralWound, .mixedTetraPak:
            self = .mixed
        case .waste, .wasteStyrofoam:
            self = .waste
        case .compost:
            self = .compost
        case .miscellaneous, .miscellaneousBackground, .miscellaneousTest:
            self = .miscellaneous
        default:
            self = .miscellaneous
        }
    }

    /// Return which submaterials are derivative from material
    var subMaterials: [SubMaterial] {
        switch self {
        case .glass:
            return [.glass, .glassBottle, .glassContainer, .glassWineBottle]
        case .metal:
            return [.metal, .metalAluminumAerosol, .metalAluminumCan, .metalAluminumFoil, .metalSteelAerosol, .metalSteelCan]
        case .paper:
            return [.paper, .paperCorrugatedCardboard, .paperEggCarton, .paperMail, .paperPaperbackBook, .paperPaperboard, .paperPaperBag, .paperPaperCup, .paperPaperTowel, .paperPizzaBox, .paperShreddedPaper]
        case .plastic:
            return [.plastic, .plasticBag, .plasticBottle, .plasticPackaging, .plasticTub, .plasticUtensil, .plasticFilm]
        case .mixed:
            return [.mixedSpiralWound, .mixedTetraPak]
        case .waste:
            return [.waste, .wasteStyrofoam]
        case .compost:
            return [.compost]
        case .miscellaneous:
            return [.miscellaneous, .miscellaneousBackground, .miscellaneousTest]
        }
    }
}
