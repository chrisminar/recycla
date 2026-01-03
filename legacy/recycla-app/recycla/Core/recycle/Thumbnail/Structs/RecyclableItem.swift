//
//  RecyclableItem.swift
//  recycla
//
//  Created by Christopher Minar on 4/21/25.
//


struct RecyclableItem {
    var material: Material?
    var subMaterial: SubMaterial?
    var category: String?
    var brand: String?
    
    init(material: Material?, subMaterial: SubMaterial?, category: String?, brand: String?, product: String?) {
        self.material = material
        self.subMaterial = subMaterial
        self.category = category
        self.brand = brand
    }
    
    /// Initilaize from only the submaterial
    init(subMaterial: SubMaterial) {
        self.material = Material(subMaterial: subMaterial)
        self.subMaterial = subMaterial
        self.category = nil
        self.brand = nil
    }
    
    /// Initialze from firebase model
    init(model: RecyclableItemModel) {
        /// initialize from sub material if possible
        let subMaterialStr: String? = model.subMaterial
        if let subMaterialStr = subMaterialStr {
            /// being a little extra verbose here to make it obvioius what is happening
            /// the "material" part of the label is kinda just a debugging output from the classifier
            /// get the true material from the submaterial
            let subMaterial = SubMaterial(snakeCase: subMaterialStr)
            self.init(subMaterial: subMaterial)
            self.category = model.category
            self.brand = model.brand
        } else {
            /// if not possible (i.e. if submatieral is none), initialize as nils
            self.init(material: nil, subMaterial: nil, category: model.category, brand: model.brand, product: nil)
        }
    }

    /// Return firebase model
    func toDataLabel() -> RecyclableItemModel {
        return RecyclableItemModel(material: material?.snakeCase ?? "", subMaterial: subMaterial?.snakeCase ?? "", category: category, brand: brand)
    }
}
