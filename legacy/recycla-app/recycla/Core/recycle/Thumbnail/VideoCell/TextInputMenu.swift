//
//  TextInputMenu.swift
//  recycla
//
//  Created by Christopher Minar on 4/8/25.
//

import SwiftUI

struct TextInputMenu: View {
    @Binding var selectedText: String?
    @State private var previousEntries: [String] = []
    @State private var isPresentingTextField = false
    @State private var newText: String = ""

    let label: String
    let placeholder: String
    let action: (String) -> Void
    var backgroundColor: Color = Color.blue
    var backgroundColorNS: Color = Color.blue.opacity(0.5)
    var systemImage: String = "arrowtriangle.down.fill"
    var defaultOptions: [String] = []

    private var userDefaultsKey: String {
        "TextInputMenuPreviousEntries_\(label)"
    }
    
    init(
        selectedText: Binding<String?>,
        label: String,
        placeholder: String,
        action: @escaping (String) -> Void
    ) {
        self._selectedText = selectedText
        self.label = label
        self.placeholder = placeholder
        self.action = action
    }

    func backgroundColor(selected: Color, notSelected: Color) -> TextInputMenu {
        var copy = self
        copy.backgroundColor = selected
        copy.backgroundColorNS = notSelected
        return copy
    }
    
    func systemImage(_ im: String) -> TextInputMenu {
        var copy = self
        copy.systemImage = im
        return copy
    }
    
    func defaultOptions(_ options: [String]) -> TextInputMenu {
        var copy = self
        copy.defaultOptions = options
        return copy
    }

    var body: some View {
        Menu {
            // Option to enter a new text string
            Button(action: {
                isPresentingTextField = true
            }) {
                Label("Enter New Text", systemImage: "square.and.pencil")
            }

            // Divider between options
            if !previousEntries.isEmpty {
                Divider()
            }

            // List of previously entered strings
            ForEach(previousEntries.reversed(), id: \.self) { entry in
                Button(action: {
                    selectedText = entry

                    // Update the order of previousEntries
                    if let index = previousEntries.firstIndex(of: entry) {
                        previousEntries.remove(at: index) // Remove the selected item
                    }
                    previousEntries.append(entry) // Re-append it to the end

                    // Save the updated list to UserDefaults
                    savePreviousEntries()

                    // Call the action with the selected value
                    action(entry)
                }) {
                    Text(entry)
                }
            }
        } label: {
            Label(selectedText ?? label, systemImage: systemImage)
                .font(.headline)
                .foregroundColor(.white)
                .frame(height: 30)
                .frame(maxWidth: .infinity)
                .background(selectedText == nil ? backgroundColorNS : backgroundColor)
                .cornerRadius(10)
                .shadow(color: backgroundColorNS, radius: 5, x: 0, y: 2)
        }
        .onAppear {
            loadPreviousEntries()
            if !["Category", "Brand", "Product", "Report"].contains(label) {
                selectedText = label
            }
            // Add defaultOptions to previousEntries
            for option in defaultOptions {
                if !previousEntries.contains(option) {
                    previousEntries.append(option)
                }
            }
            // Ensure the list does not exceed 10 entries
            if previousEntries.count > 10 {
                previousEntries = Array(previousEntries.suffix(10))
            }
        }
        .sheet(isPresented: $isPresentingTextField) {
            VStack {
                Text("Enter New \(label)")
                    .font(.headline)
                    .padding()

                TextField(placeholder, text: $newText)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .padding()

                HStack {
                    Button("Cancel") {
                        isPresentingTextField = false
                    }
                    .padding()
                    .foregroundColor(.red)

                    Spacer()

                    Button("Save") {
                        if (!newText.isEmpty) {
                            selectedText = newText

                            // Remove the value if it already exists
                            if let index = previousEntries.firstIndex(of: newText) {
                                previousEntries.remove(at: index)
                            }

                            // Append the new value to the end
                            previousEntries.append(newText)

                            // Ensure the list does not exceed 10 entries
                            if previousEntries.count > 10 {
                                previousEntries.removeFirst()
                            }

                            // Save the updated list to UserDefaults
                            savePreviousEntries()

                            // Call the action with the new value
                            action(newText)

                            // Clear the input field
                            newText = ""
                        }
                        isPresentingTextField = false
                    }
                    .padding()
                    .foregroundColor(newText.count >= 3 ? .blue : .gray) // Change color based on character count
                }
            }
            .padding()
        }
    }

    // MARK: - Persistence Methods
    private func loadPreviousEntries() {
        if let savedEntries = UserDefaults.standard.array(forKey: userDefaultsKey) as? [String] {
            previousEntries = savedEntries
        }
    }

    private func savePreviousEntries() {
        UserDefaults.standard.set(previousEntries, forKey: userDefaultsKey)
    }
}

#Preview {
//    @State @Previewable var selectedText: String? = nil
    @State @Previewable var report: String? = nil
//    TextInputMenu(selectedText: $selectedText, label: "Fruit", placeholder: "New Fruit", action: { text in
//        print("Selected text: \(text)")
//    })
    TextInputMenu(
        selectedText: $report,
        label: "Report",
        placeholder: "What went wrong?",
        action: { text in
            print("Selected text: \(text)")
        }
    )
    .backgroundColor(selected: Color.red, notSelected: Color.red.opacity(0.5))
    .systemImage("exclamationmark.triangle.fill")
    .defaultOptions(["Default1"])
}
