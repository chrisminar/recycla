//
//  RecycleGroup.swift
//  recycla
//
//  Created by Christopher Minar on 4/21/25.
//

/// How recylable things are
/// 1 = always (not actully always but common)
/// 5 = never
enum RecycleGroup: String {
    case glassBottlesAndJars = "Glass bottles & jars"
    case aluminumTrays = "Aluminum trays"
    case aluminumAerosolContainers = "Aluminum aerosol containers"
    case aluminumCans = "Aluminum cans"
    case aluminumFoil = "Aluminum foil"
    case steelTinBimetalCans = "Steel/tin/bimetal cans"
    case steelAerosolContainers = "Steel aerosol containers"
    case nonmetalizedWrappingPaper = "Nonmetalized wrapping paper"
    case corrugatedCardboard = "Corrugated cardboard"
    case paperboard = "Paperboard"
    case polycoatedPaperboard = "Polycoated paperboard"
    case paperBags = "Paper bags"
    case paperHotCups = "Paper hot cups"
    case waste = "Waste"
    case pizzaBoxes = "Pizza boxes"
    case shreddedPaper = "Shredded paper"
    case magazines = "Magazines"
    case mailOfficePaper = "Mail/office paper"
    case newspaper = "Newspaper"
    case paperbackBooks = "Paperback books"
    case petBottlesJugs = "PET bottles/jugs"
    case hdpeBottlesJugsJars = "HDPE bottles/jugs & jars"
    case ldpeBottlesJugsJars = "LDPE bottles/jugs & jars"
    case ppBottlesJugsJars = "PP bottles/jugs & jars"
    case ppClamshells = "PP clamshells"
    case petThermoforms = "PET thermoforms"
    case petCups = "PET cups"
    case hdpeCups = "HDPE cups"
    case ldpeTubsCups = "LDPE tubs/cups"
    case ppCups = "PP cups"
    case petLids = "PET lids"
    case hdpeLids = "HDPE lids"
    case ldpeLids = "LDPE lids"
    case ppLids = "PP lids"
    case psLids = "PS lids"
    case monomaterialPeBagsAndFilm = "Monomaterial PE bags and film"
    case otherPlasticPackaging = "Other plastic packaging"
    case psFoodservicePackaging = "PS foodservice packaging"
    case epsFoodservicePackaging = "EPS foodservice packaging"
    case epsPackaging = "EPS packaging"
    case pvcPackaging = "PVC packaging"
    case multimaterialFlexiblePackaging = "Multimaterial flexible packaging"
    case moldedFiberFoodservicePackaging = "Molded fiber foodservice packaging"
    case moldedFiberNonfoodPackagingAndTrays = "Molded fiber nonfood packaging and trays"
    case hdpeTrays = "HDPE trays"
    case hdpeTubes = "HDPE tubes"
    case hdpeBulkyRigidPlastics = "HDPE bulky rigid plastics"
    case ppTrays = "PP trays"
    case ppBulkyRigidPlastics = "PP bulky rigid plastics"
    case ppPods = "PP pods"
    case ppTubes = "PP tubes"
    case plasticBuckets = "Plastic buckets (≥ 3 gal)"
    case nurseryPlantPackaging = "Nursery (plant) packaging"
    case hdpeTubs = "HDPE tubs"
    case ppTubs = "PP tubs"
    case spiralWoundContainers = "Spiral wound containers"
    case asepticGableTopCartons = "Aseptic/gable-top cartons"
    case other = "Other"

    var intValue: Int {
        switch self {
        case .glassBottlesAndJars: return 1
        case .aluminumTrays: return 1
        case .aluminumAerosolContainers: return 2
        case .aluminumCans: return 1
        case .aluminumFoil: return 2
        case .steelTinBimetalCans: return 1
        case .steelAerosolContainers: return 2
        case .nonmetalizedWrappingPaper: return 2
        case .corrugatedCardboard: return 1
        case .paperboard: return 1
        case .polycoatedPaperboard: return 2
        case .paperBags: return 2
        case .paperHotCups: return 2
        case .waste: return 4
        case .pizzaBoxes: return 2
        case .shreddedPaper: return 2
        case .magazines: return 1
        case .mailOfficePaper: return 1
        case .newspaper: return 1
        case .paperbackBooks: return 2
        case .petBottlesJugs: return 1
        case .hdpeBottlesJugsJars: return 1
        case .ldpeBottlesJugsJars: return 2
        case .ppBottlesJugsJars: return 2
        case .ppClamshells: return 2
        case .petThermoforms: return 2
        case .petCups: return 2
        case .hdpeCups: return 2
        case .ldpeTubsCups: return 2
        case .ppCups: return 2
        case .petLids: return 2
        case .hdpeLids: return 2
        case .ldpeLids: return 2
        case .ppLids: return 2
        case .psLids: return 3
        case .monomaterialPeBagsAndFilm: return 3
        case .otherPlasticPackaging: return 3
        case .psFoodservicePackaging: return 3
        case .epsFoodservicePackaging: return 3
        case .epsPackaging: return 3
        case .pvcPackaging: return 3
        case .multimaterialFlexiblePackaging: return 3
        case .moldedFiberFoodservicePackaging: return 3
        case .moldedFiberNonfoodPackagingAndTrays: return 3
        case .hdpeTrays: return 2
        case .hdpeTubes: return 3
        case .hdpeBulkyRigidPlastics: return 2
        case .ppTrays: return 2
        case .ppBulkyRigidPlastics: return 2
        case .ppPods: return 3
        case .ppTubes: return 3
        case .plasticBuckets: return 2
        case .nurseryPlantPackaging: return 2
        case .hdpeTubs: return 2
        case .ppTubs: return 2
        case .spiralWoundContainers: return 3
        case .asepticGableTopCartons: return 2
        case .other: return 5
        }
    }
}
