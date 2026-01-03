//
//  ThumbnailView.swift
//  recycla
//
//  Created by Christopher Minar on 2/6/25.
//


import Foundation
import SwiftUI
import FirebaseFirestore


struct ThumbnailView: View {
    
    @StateObject var viewModel = StateManager.shared
    @ObservedObject var thumbnailManager = ThumbnailManager.shared
    
    var body: some View {
        VStack {
            HStack {
                Spacer()
                Menu {
                    ForEach($thumbnailManager.menuOptions) { $option in
                        Button(role: nil) { // Prevent menu dismissal
                            option.isChecked.toggle()
                        } label: {
                            Label(option.label, systemImage: option.isChecked ? "checkmark.square" : "square")
                        }
                    }
                } label: {
                    Label("Filters", systemImage: "arrow.3.trianglepath")
                        .padding()
                }
                Menu {
                    ForEach($thumbnailManager.timeFilterOptions) { $option in
                        Button {
                            thumbnailManager.selectTimeFilter(option: option)
                        } label: {
                            Label(option.label, systemImage: option.isSelected ? "checkmark.square" : "square")
                        }
                    }
                } label: {
                    Label("Date", systemImage: "calendar")
                        .padding()
                }
            }
            .padding(.trailing)
            
            ScrollView {
                LazyVStack {
                    if viewModel.userVideos.isEmpty {
                        NoDataView()
                    } else {
                        ForEach(viewModel.userVideos.reversed(), id: \.self) { videoId in
                            VideoCellView(videoId: videoId)
                        }
                    }
                }
                .onFirstAppear {
                    viewModel.addListenerForVideos()
                }
            }
        }
    }
}

struct NoDataView: View{
    var body: some View {
        Text("No data available, please recycle some items to see your progress!")
            .padding()
            .background(.background)
            .cornerRadius(10)
            .shadow(color: shadowColor, radius: 5)
            .padding()
    }

    private var shadowColor: Color {
        Color(UIColor { traitCollection in
            traitCollection.userInterfaceStyle == .dark ? UIColor.white.withAlphaComponent(0.5) :
                UIColor.black.withAlphaComponent(0.5)
        })
    }
}

#Preview {
    ThumbnailView()
}
