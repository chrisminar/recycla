//
//  ExpandableRowView.swift
//  recycla
//
//  Created by Christopher Minar on 4/3/25.
//

import SwiftUI

struct ExpandableRowView<Content: View>: View {
    let title: String
    let content: () -> Content // Accept content as a closure
    @State private var isExpanded: Bool
    @State private var bounce: Bool = false

    init(title: String, isExpanded: Bool = false, @ViewBuilder content: @escaping () -> Content) {
        self.title = title
        self._isExpanded = State(initialValue: isExpanded) // Initialize @State with the provided value
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading) {
            Button(action: {
                withAnimation {
                    isExpanded.toggle()
                }
            }) {
                HStack {
                    Text(title)
                        .font(.headline)
                        .foregroundColor(.primary)
                    Spacer()
                    Image(systemName: "chevron.right")
                        .foregroundColor(.gray)
                        .rotationEffect(.degrees(isExpanded ? 90 : 0)) // Rotate when expanded
                        .animation(.easeInOut(duration: 0.2), value: isExpanded) // Smooth rotation animation
                        .offset(x: bounce ? -2 : 2) // Bounce effect
                        .animation(.spring(response: 1, dampingFraction: 0.1, blendDuration: 2.5), value: bounce)
                }
                .padding()
            }

            if isExpanded {
                content() // Call the closure to render the content
                    .padding(.leading)
                    .transition(.opacity.combined(with: .slide))
            }
        }
        .onAppear {
            // Start a timer to trigger the bounce animation periodically
            Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { _ in
                if !isExpanded {
                    withAnimation {
                        bounce.toggle()
                    }
                }
            }
        }
    }
}

#Preview {
    ExpandableRowView(title: "Expandable Row", isExpanded: false) {
        Text("This is the expanded content.")
    }
}
