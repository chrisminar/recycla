//
//  SubMaterial.swift
//  recycla
//
//  Created by Christopher Minar on 4/21/25.
//

import Foundation

enum SubMaterial: String, CaseIterable {
    case glass = "glass"
    case glassBottle = "glassBottle"
    case glassContainer = "glassContainer"
    case glassWineBottle = "glassWineBottle"
    case metal = "metal"
    case metalAluminumAerosol = "metalAluminumAerosol"
    case metalAluminumCan = "metalAluminumCan"
    case metalAluminumFoil = "metalAluminumFoil"
    case metalSteelAerosol = "metalSteelAerosol"
    case metalSteelCan = "metalSteelCan"
    case paper = "paper"
    case paperCorrugatedCardboard = "paperCorrugatedCardboard"
    case paperEggCarton = "paperEggCarton"
    case paperMail = "paperMail"
    case paperPaperbackBook = "paperPaperbackBook"
    case paperPaperboard = "paperPaperboard"
    case paperPaperBag = "paperPaperBag"
    case paperPaperCup = "paperPaperCup"
    case paperPaperTowel = "paperPaperTowel"
    case paperPizzaBox = "paperPizzaBox"
    case paperShreddedPaper = "paperShreddedPaper"
    case plastic = "plastic"
    case plasticFilm = "plasticFilm"
    case plasticBag = "plasticBag"
    case plasticBottle = "plasticBottle"
    case plasticPackaging = "plasticPackaging"
    case plasticTub = "plasticTub"
    case plasticUtensil = "plasticUtensil"
    case mixedSpiralWound = "mixedSpiralWound"
    case mixedTetraPak = "mixedTetraPak"
    case waste = "waste"
    case wasteStyrofoam = "wasteStyrofoam"
    case compost = "compost"
    case miscellaneous = "miscellaneous"
    case miscellaneousBackground = "miscellaneousBackground"
    case miscellaneousTest = "miscellaneousTest"

    /// Get snake case (python/firebase) version of enum
    var snakeCase: String {
        switch self {
        case .glass: return "glass"
        case .glassBottle: return "glass, bottle"
        case .glassContainer: return "glass, container"
        case .glassWineBottle: return "glass, wine_bottle"
        case .metal: return "metal"
        case .metalAluminumAerosol: return "metal, aluminum_aerosol"
        case .metalAluminumCan: return "metal, aluminum_can"
        case .metalAluminumFoil: return "metal, aluminum_foil"
        case .metalSteelAerosol: return "metal, steel_aerosol"
        case .metalSteelCan: return "metal, steel_can"
        case .paper: return "paper"
        case .paperCorrugatedCardboard: return "paper, corrugated_cardboard"
        case .paperEggCarton: return "paper, egg_carton"
        case .paperMail: return "paper, mail"
        case .paperPaperbackBook: return "paper, paperback_book"
        case .paperPaperboard: return "paper, paperboard"
        case .paperPaperBag: return "paper, paper_bag"
        case .paperPaperCup: return "paper, paper_cup"
        case .paperPaperTowel: return "paper, paper_towel"
        case .paperPizzaBox: return "paper, pizza_box"
        case .paperShreddedPaper: return "paper, shredded_paper"
        case .plastic: return "plastic"
        case .plasticBag: return "plastic, bag"
        case .plasticBottle: return "plastic, bottle"
        case .plasticPackaging: return "plastic, packaging"
        case .plasticFilm: return "plastic, film"
        case .plasticTub: return "plastic, tub"
        case .plasticUtensil: return "plastic, utensil"
        case .mixedSpiralWound: return "mixed, spiral_wound"
        case .mixedTetraPak: return "mixed, tetra_pak"
        case .waste: return "waste"
        case .wasteStyrofoam: return "waste, styrofoam"
        case .compost: return "compost"
        case .miscellaneous: return "miscellaneous"
        case .miscellaneousBackground: return "miscellaneous, background"
        case .miscellaneousTest: return "miscellaneous, test"
        }
    }
    
    /// Get snake case (python/firebase) version of enum
    var pretty: String {
        switch self {
        case .glass: return "Glass"
        case .glassBottle: return "Glass Bottle"
        case .glassContainer: return "Glass Container"
        case .glassWineBottle: return "Wine Bottle"
        case .metal: return "Metal"
        case .metalAluminumAerosol: return "Aluminum Aerosol"
        case .metalAluminumCan: return "Aluminum Can"
        case .metalAluminumFoil: return "Aluminum Foil"
        case .metalSteelAerosol: return "Steel Aerosol"
        case .metalSteelCan: return "Steel Can"
        case .paper: return "Paper"
        case .paperCorrugatedCardboard: return "Corrugated Cardboard"
        case .paperEggCarton: return "Egg Carton"
        case .paperMail: return "Mail"
        case .paperPaperbackBook: return "Paperback Book"
        case .paperPaperboard: return "Paperboard"
        case .paperPaperBag: return "Paper Bag"
        case .paperPaperCup: return "Paper Cup"
        case .paperPaperTowel: return "Paper Towel"
        case .paperPizzaBox: return "Pizza Box"
        case .paperShreddedPaper: return "Shredded Paper"
        case .plastic: return "Plastic"
        case .plasticBag: return "Plastic Bag"
        case .plasticBottle: return "Plastic Bottle"
        case .plasticPackaging: return "Plastic Packaging"
        case .plasticTub: return "Plastic Tub"
        case .plasticFilm: return "Plastic Film"
        case .plasticUtensil: return "Utensil"
        case .mixedSpiralWound: return "Spiral Wound Canister"
        case .mixedTetraPak: return "Tetra Pak"
        case .waste: return "waste"
        case .wasteStyrofoam: return "Styrofoam"
        case .compost: return "Compost"
        case .miscellaneous: return "Miscellaneous"
        case .miscellaneousBackground: return "Background"
        case .miscellaneousTest: return "Test"
        }
    }
    
    /// Should the category/brand/product GT selection be displayed
    var brandIsInteresting: Bool {
        switch self {
        case .glass: return false
        case .glassBottle: return true
        case .glassContainer: return true
        case .glassWineBottle: return true
        case .metal: return false
        case .metalAluminumAerosol: return true
        case .metalAluminumCan: return true
        case .metalAluminumFoil: return false
        case .metalSteelAerosol: return true
        case .metalSteelCan: return true
        case .paper: return false
        case .paperCorrugatedCardboard: return true
        case .paperEggCarton: return false
        case .paperMail: return false
        case .paperPaperbackBook: return false
        case .paperPaperboard: return true
        case .paperPaperBag: return false
        case .paperPaperCup: return false
        case .paperPaperTowel: return false
        case .paperPizzaBox: return false
        case .paperShreddedPaper: return false
        case .plastic: return true
        case .plasticBag: return false
        case .plasticFilm: return false
        case .plasticBottle: return true
        case .plasticPackaging: return true
        case .plasticTub: return true
        case .plasticUtensil: return false
        case .mixedSpiralWound: return true
        case .mixedTetraPak: return true
        case .waste: return false
        case .wasteStyrofoam: return false
        case .compost: return false
        case .miscellaneous: return false
        case .miscellaneousBackground: return false
        case .miscellaneousTest: return false
        }
    }
    
    /// Get the snake case sub material from the full string
    var justSub: String {
        if let range = snakeCase.range(of: ", ") {
            let subString = String(snakeCase[range.upperBound...])
            return subString.replacingOccurrences(of: "_", with: " ").capitalized
        }
        return ""
    }
    
    /// Is this submaterial recyclable?
    var isRecyclable: (Int, Bool, String) {
        let uniqueRecycleGroups = self.recycleGroupIntValues
        let ambiguous = uniqueRecycleGroups.count > 1
        let recycleGroup = uniqueRecycleGroups.first ?? 0
        let recyclabilityString = recycleGroup == 0 ? "" : self.recyclability
        return (recycleGroup, ambiguous, recyclabilityString)
    }

    /// Get recycle groups for this enum
    var recycleGroupIntValues: [Int] {
        // Get unique recycle Group ints for each classifer label
        return Array(Set(recycleGroups.map { $0.intValue })).sorted()
    }

    /// Recycle Groups for each submaterial
    var recycleGroups: [RecycleGroup] {
        switch self {
        case .glass:
            return [.glassBottlesAndJars]
        case .glassBottle, .glassContainer, .glassWineBottle:
            return [.glassBottlesAndJars]
        case .metal:
            return [.aluminumTrays]
        case .metalAluminumAerosol:
            return [.aluminumAerosolContainers]
        case .metalAluminumCan:
            return [.aluminumCans]
        case .metalAluminumFoil:
            return [.aluminumFoil]
        case .metalSteelCan:
            return [.steelTinBimetalCans]
        case .metalSteelAerosol:
            return [.steelAerosolContainers]
        case .paper:
            return [.mailOfficePaper, .nonmetalizedWrappingPaper, .magazines, .newspaper]
        case .paperCorrugatedCardboard:
            return [.corrugatedCardboard]
        case .paperEggCarton:
            return [.paperboard]
        case .paperMail:
            return [.mailOfficePaper]
        case .paperPaperbackBook:
            return [.paperbackBooks]
        case .paperPaperboard:
            return [.paperboard, .polycoatedPaperboard]
        case .paperPaperBag:
            return [.paperBags]
        case .paperPaperCup:
            return [.paperHotCups]
        case .paperPaperTowel:
            return [.waste]
        case .paperPizzaBox:
            return [.pizzaBoxes, .waste]
        case .paperShreddedPaper:
            return [.shreddedPaper]
        case .plastic:
            return [.petBottlesJugs, .hdpeTrays, .hdpeTubes, .hdpeBulkyRigidPlastics, .ppTrays, .ppBulkyRigidPlastics, .ppPods, .ppTubes, .plasticBuckets, .nurseryPlantPackaging, .ppClamshells, .petThermoforms, .petCups, .hdpeCups, .ldpeTubsCups, .ppCups, .petLids, .hdpeLids, .ldpeLids, .ppLids, .psLids]
        case .plasticBag:
            return [.monomaterialPeBagsAndFilm]
        case .plasticFilm:
            return [.monomaterialPeBagsAndFilm, .otherPlasticPackaging]
        case .plasticBottle:
            return [.petBottlesJugs, .hdpeBottlesJugsJars, .ldpeBottlesJugsJars, .ppBottlesJugsJars]
        case .plasticPackaging:
            return [.monomaterialPeBagsAndFilm, .otherPlasticPackaging, .psFoodservicePackaging, .epsFoodservicePackaging, .epsPackaging, .pvcPackaging, .multimaterialFlexiblePackaging, .moldedFiberFoodservicePackaging, .moldedFiberNonfoodPackagingAndTrays]
        case .plasticTub:
            return [.hdpeTubs, .ppTubs]
        case .plasticUtensil:
            return [.petThermoforms]
        case .mixedSpiralWound:
            return [.spiralWoundContainers]
        case .mixedTetraPak:
            return [.asepticGableTopCartons]
        case .waste, .wasteStyrofoam:
            return [.waste]
        case .compost:
            return [.waste]
        case .miscellaneous, .miscellaneousBackground, .miscellaneousTest:
            return [.other]
        }
    }

    /// Human readable string for how recyclable each submaterial is.
    var recyclability: String {
        switch self {
        case .glass, .glassBottle, .glassContainer, .glassWineBottle:
            return "Almost all glass containers are recyclable if clean."
        case .metal:
            return "Most metal is recyclable if clean."
        case .metalAluminumAerosol, .metalSteelAerosol:
            return "Aerosols are recyclable if empty."
        case .metalAluminumCan, .metalSteelCan:
            return "Metal cans are recyclable if clean."
        case .metalAluminumFoil:
            return "Aluminum foil is recyclable if it's clean."
        case .paper, .paperMail:
            return "Most paper producets are recyclable."
        case .paperCorrugatedCardboard:
            return "Cardboard is recyclable."
        case .paperEggCarton:
            return "Egg cartons are recyclable."
        case .paperPaperbackBook:
            return "Paperback books are recyclable."
        case .paperPaperboard:
            return "Paperboard is typically recyclable. If it's polycoated (water resistant), it might be recyclable."
        case .paperPaperBag:
            return "Paper bags are recyclable. If they are plastic-lined, they might not be recyclable."
        case .paperPaperCup:
            return "Paper cups are sometimes recyclable if clean. The plastic lining makes them difficult to recycle."
        case .paperPaperTowel:
            return "Paper towels are not recyclable, even if they are clean. Consider compost."
        case .paperPizzaBox:
            return "Pizza boxes are recyclable if they are clean. If they are greasy, consider compost."
        case .paperShreddedPaper:
            return "Shredded paper is recyclable, but might not be accepted in curbside recycling due to processing difficulty."
        case .plastic, .plasticBag, .plasticBottle, .plasticPackaging, .plasticTub, .plasticUtensil, .plasticFilm:
            return "Check plastic recycling code. 1 & 2 are usually recyclable. (3,4,5) are sometimes recyclable. 6 is rearely recyclable. 7 is usually not recyclable. Remove lids and clean before recycling."
        case .mixedTetraPak:
            return "Tetra Paks are sometimes recyclable."
        case .mixedSpiralWound:
            return "Spiral wound canisters are rarely recyclable."
        case .waste, .wasteStyrofoam, .compost:
            return "Not recyclable."
        case .miscellaneous, .miscellaneousBackground, .miscellaneousTest:
            return "Not disposable?"
        }
    }

    /// Initialize from snake case
    init(snakeCase: String) {
        switch snakeCase {
        case "glass": self = .glass
        case "glass, bottle": self = .glassBottle
        case "glass, container": self = .glassContainer
        case "glass, wine_bottle": self = .glassWineBottle
        case "metal": self = .metal
        case "metal, aluminum_aerosol": self = .metalAluminumAerosol
        case "metal, aluminum_can": self = .metalAluminumCan
        case "metal, aluminum_foil": self = .metalAluminumFoil
        case "metal, steel_aerosol": self = .metalSteelAerosol
        case "metal, steel_can": self = .metalSteelCan
        case "paper": self = .paper
        case "paper, corrugated_cardboard": self = .paperCorrugatedCardboard
        case "paper, egg_carton": self = .paperEggCarton
        case "paper, mail": self = .paperMail
        case "paper, paperback_book": self = .paperPaperbackBook
        case "paper, paperboard": self = .paperPaperboard
        case "paper, paper_bag": self = .paperPaperBag
        case "paper, paper_cup": self = .paperPaperCup
        case "paper, paper_towel": self = .paperPaperTowel
        case "paper, pizza_box": self = .paperPizzaBox
        case "paper, shredded_paper": self = .paperShreddedPaper
        case "plastic": self = .plastic
        case "plastic, bag": self = .plasticBag
        case "plastic, bottle": self = .plasticBottle
        case "plastic, packaging": self = .plasticPackaging
        case "plastic, tub": self = .plasticTub
        case "plastic, utensil": self = .plasticUtensil
        case "plastic, film": self = .plasticFilm
        case "mixed, spiral_wound": self = .mixedSpiralWound
        case "mixed, tetra_pak": self = .mixedTetraPak
        case "waste": self = .waste
        case "waste, styrofoam": self = .wasteStyrofoam
        case "compost": self = .compost
        case "miscellaneous": self = .miscellaneous
        case "miscellaneous, background": self = .miscellaneousBackground
        case "miscellaneous, test": self = .miscellaneousTest
        default: self = .miscellaneous
        }
    }
    
    /// The classifier can't produce every label yet. This gets the closest label (what the classifier considers) the sub material
    var nearestLabel: SubMaterial {
        switch self {
        case .glass: return .glass
        case .glassBottle: return .glass
        case .glassContainer: return .glass
        case .glassWineBottle: return .glassWineBottle
        case .metal: return .metal
        case .metalAluminumAerosol: return .metal
        case .metalAluminumCan: return .metalAluminumCan
        case .metalAluminumFoil: return .metalAluminumFoil
        case .metalSteelAerosol: return .metal
        case .metalSteelCan: return .metalSteelCan
        case .paper: return .paper
        case .paperCorrugatedCardboard: return .paperCorrugatedCardboard
        case .paperEggCarton: return .paperEggCarton
        case .paperMail: return .paper
        case .paperPaperbackBook: return .paper
        case .paperPaperboard: return .paperPaperboard
        case .paperPaperBag: return .paper
        case .paperPaperCup: return .paperPaperCup
        case .paperPaperTowel: return .paperPaperTowel
        case .paperPizzaBox: return .paperCorrugatedCardboard
        case .paperShreddedPaper: return .paper
        case .plastic: return .plastic
        case .plasticBag: return .plasticFilm
        case .plasticFilm: return .plasticFilm
        case .plasticBottle: return .plasticBottle
        case .plasticPackaging: return .plasticFilm
        case .plasticTub: return .plastic
        case .plasticUtensil: return .plastic
        case .mixedSpiralWound: return .mixedSpiralWound
        case .mixedTetraPak: return .mixedTetraPak
        case .waste: return .waste
        case .wasteStyrofoam: return .waste
        case .compost: return .compost
        case .miscellaneous: return .miscellaneous
        case .miscellaneousBackground: return .miscellaneousBackground
        case .miscellaneousTest: return .miscellaneousTest
        }
    }
    
}
