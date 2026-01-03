//
//  TimeSinceView.swift
//  recycla
//
//  Created by Christopher Minar on 4/2/25.
//

import SwiftUI

struct TimeSinceView: View {
    let uploadDate: Date
    @State private var timer: Timer?
    @State private var currentTime: Date = Date()

    var body: some View {
        Text(Utilities.shared.timeSince(uploadDate))
            .id(currentTime)
            .onAppear {
                timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
                    currentTime = Date()
                }
            }
            .onDisappear {
                timer?.invalidate()
            }
    }
}

#Preview {
    TimeSinceView(uploadDate: Date())
}
