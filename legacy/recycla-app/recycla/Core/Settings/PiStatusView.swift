//
//  PiStatusView.swift
//  recyclo-app
//
//  Created by Christopher Minar on 3/7/25.
//
import SwiftUI

// TODO/WARNING. This will show piids, but the status dot will be duplicated from the first pi to the others
struct PiStatusView: View {
    @ObservedObject private var piManager = PiManager.shared
    var piIds: [String]?
    @State private var timer: Timer?
    @State private var showPopup = false
    @State private var popupMessage = ""

    var body: some View {
        ZStack {
            VStack {
                if let piIds = piIds, !piIds.isEmpty {
                    ForEach(piIds, id: \.self) { piId in
                        HStack {
                            let connectionColor = piManager.connectionStatus
                            Image(systemName: "circle.fill")
                                .foregroundColor(connectionColor.color)
                                .onLongPressGesture(minimumDuration: 0.5) {
                                    popupMessage = connectionColor.description
                                    showPopup = true
                                } onPressingChanged: { pressing in
                                    if !pressing {
                                        showPopup = false
                                    }
                                }
                            Text("  \(showPopup ? popupMessage : piManager.yourPiNickname)")
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }.onAppear {
                        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
                            piManager.objectWillChange.send()
                        }
                    }
                    .onDisappear {
                        timer?.invalidate()
                    }
                } else {
                    ProgressView()
                }
            }
        }
    }
}

#Preview {
    PiStatusView()
}
