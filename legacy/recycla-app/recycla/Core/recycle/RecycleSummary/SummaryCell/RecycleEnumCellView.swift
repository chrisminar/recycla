//
//  RecycleEnumCellView.swift
//  recycla
//
//  Created by Christopher Minar on 2/5/25.
//

import SwiftUI

struct RecycleEnumCellView: View {
    
    let itemName: String?
    let numRecycle: Int?
    let accuracy: Double?
    
    init(itemName: String? = nil, numRecycle: Int? = nil, accuracy: Double? = nil) {
        self.itemName = itemName
        self.numRecycle = numRecycle
        self.accuracy = accuracy
    }
    
    var body: some View {
        GeometryReader { geometry in
            HStack{
                Text(itemName.map { "\($0)" } ?? "Class")
                    .font(.title3)
                    .padding(.leading, 5)
                    .frame(width: geometry.size.width * 0.4, alignment: .leading)
                Text(numRecycle.map { "\($0)" } ?? "Count")
                    .font(.headline)
                    .frame(width: geometry.size.width * 0.2, alignment: .center)
                Text(accuracy.map { "\($0, specifier: "%0.1f")" } ?? "Accuracy")
                    .font(.headline)
                    .padding(.trailing, 5)
                    .frame(width: geometry.size.width * 0.3, alignment: .trailing)
            }
            .frame(maxWidth: .infinity)
            .frame(height: 35)
        }
        .frame(maxWidth: .infinity)
        .frame(height: 30)
        .padding(.leading, 10)
        .padding(.trailing, 10)
    }
}

#Preview {
    VStack{
        RecycleEnumCellView()
        Divider()
        RecycleEnumCellView(itemName:"My class", numRecycle: 5, accuracy:4/5)
        RecycleEnumCellView(itemName:"Second class", numRecycle: 2, accuracy:1/2)
    }
}
