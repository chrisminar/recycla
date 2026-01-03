//
//  RecycleEnumView.swift
//  recycla
//
//  Created by Christopher Minar on 2/5/25.
//

import SwiftUI


struct RecycleEnumView: View {
    @Binding var num: [String: Int]
    @Binding var accuracy: [String: Double]
    let isSub: Bool
        
    var body: some View {
        VStack{
             RecycleEnumCellView()
             Divider()
             if !num.isEmpty && !accuracy.isEmpty {
                 ScrollView{
                     VStack{
                         ForEach(num.keys.sorted(by: { num[$0]! > num[$1]! }), id: \.self) { key in
                             if isSub {
                                 let name = SubMaterial(snakeCase: key).pretty
                                 RecycleEnumCellView(itemName: name, numRecycle: num[key], accuracy: accuracy[key])
                             } else {
                                 RecycleEnumCellView(itemName: key, numRecycle: num[key], accuracy: accuracy[key])
                             }
                         }
                     }
                 }
                 .fixedSize(horizontal: false, vertical: true)
                 
            } else {
                NoDataView()
            }
        }
    }
}

#Preview {
    @State @Previewable var num: [String:Int] = [:]
    @State @Previewable var acc: [String:Double] = [:]
    @State @Previewable var num2: [String:Int] = ["Plastic": 5, "Glass": 3, "Metal": 8]
    @State @Previewable var acc2: [String:Double] = ["Plastic": 60, "Glass": 100, "Metal": 100]
    @State @Previewable var num3: [String:Int] = ["Plastic": 4, "Plastic, PlasticBottle":1, "Glass": 3, "Metal":8]
    @State @Previewable var acc3: [String:Double] = ["Plastic": 75, SubMaterial.plasticBottle.snakeCase:0, "Glass": 100, "Metal":100]
    VStack {
        RecycleEnumView(num: $num, accuracy: $acc, isSub: false)
        Divider()
        RecycleEnumView(num: $num2, accuracy: $acc2, isSub: false)
        Divider()
        RecycleEnumView(num: $num3, accuracy: $acc3, isSub: true)
        Spacer()
    }
}
