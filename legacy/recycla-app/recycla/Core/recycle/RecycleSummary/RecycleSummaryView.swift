//
//  RecycleSummaryView.swift
//  recycla
//
//  Created by Christopher Minar on 2/5/25.
//

import SwiftUI

@MainActor
final class SummaryManager: BaseSummaryManager {
    static let shared = SummaryManager()
    private override init() {
        super.init()
        loadFilters()
    }

    override func selectTimeFilter(_ period: SummaryPeriod) {
        self.timeFilterOption = period
        StateManager.shared.setSummaryPeriod(period, statsType: .summary)
        saveFilters()
    }

    override func saveFilters() {
        UserDefaults.standard.set(timeFilterOption.rawValue, forKey: "timeFilterOptionsSummary")
    }

    override func loadFilters() {
        if let savedPeriodRaw = UserDefaults.standard.string(forKey: "timeFilterOptionsSummary"),
           let savedPeriod = SummaryPeriod(rawValue: savedPeriodRaw) {
            selectTimeFilter(savedPeriod)
        } else {
            selectTimeFilter(.all)
        }
    }
}


struct FilterButtonView: View {
    @StateObject var summaryManager = SummaryManager.shared
    
    var body: some View {
        Menu {
            ForEach(SummaryPeriod.allCases, id: \.self) { period in
                Button {
                    summaryManager.selectTimeFilter(period)
                } label: {
                    Label(period.rawValue.capitalized, systemImage: summaryManager.timeFilterOption == period ? "checkmark.square" : "square")
                }
            }
        } label: {
            Label("Date", systemImage: "calendar")
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color(.systemGray6))
                .cornerRadius(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color(.systemGray4), lineWidth: 0.5)
                )
        }
        
    }
}

struct RecycleSummaryView: View {
    @StateObject var stateShared = StateManager.shared
    @StateObject var summaryManager = SummaryManager.shared

    var body: some View {
        NavigationStack {
            ScrollView {
                // time filter menu
                VStack {
                    HStack {
                        Spacer()
                        // Show period text if not empty
                        if !summaryManager.periodText().isEmpty {
                            Text(summaryManager.periodText())
                                .font(.subheadline)
                                .foregroundColor(.secondary)
                        }
                        FilterButtonView()
                    }
                    .padding(.bottom, 4)

                    ExpandableRowView(title: "Item summary", isExpanded: true) {
                        ItemsRecycledView()
                    }

                    Divider()
                    
                    ExpandableRowView(title: "Weekly Items Recycled", isExpanded: true) {
                        WeeklyPlotView()
                    }
                    
                    Divider()

                    ExpandableRowView(title: "Material Breakdown", isExpanded: true) {
                        RecycleEnumView(num: $stateShared.summaryStats.materialNumClass, accuracy: $stateShared.summaryStats.materialPctClassCorrect, isSub: false)
                    }

                    Divider()

                    ExpandableRowView(title: "Category Breakdown") {
                        RecycleEnumView(num: $stateShared.summaryStats.subMaterialNumClass, accuracy: $stateShared.summaryStats.subMaterialPctClassCorrect, isSub: true)
                    }
                    
                    Spacer()
                }
                .onFirstAppear {
                    stateShared.addListenerSummary(period: stateShared.currentSummaryPeriod, statsType: .summary)
                }
            }
        }
        .padding()
        .navigationTitle("Summary")
    }
}

struct ItemsRecycledView: View {
    @StateObject var stateShared = StateManager.shared

    var body: some View {
        VStack {
            HStack {
                Text("#Items")
                
                Spacer()

                Text("Material\nAccuracy")
                    .multilineTextAlignment(.center)

                Spacer()

                Text("Submaterial\nAccuracy")
                    .multilineTextAlignment(.trailing)
            }
            .font(.headline)

            Divider()

            HStack {
                Text("\(stateShared.summaryStats.numItems)")
                
                Spacer()

                Text("\(stateShared.summaryStats.materialPctCorrect, specifier: "%.1f")%")

                Spacer()

                Text("\(stateShared.summaryStats.subMaterialPctCorrect, specifier: "%.1f")%")
            }
            .font(.headline)
        }
    }
}

#Preview {
    NavigationStack {
        RecycleSummaryView()
    }
}
