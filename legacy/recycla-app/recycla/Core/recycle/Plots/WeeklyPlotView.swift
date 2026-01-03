//
//  WeeklyPlotView.swift
//  recycla
//
//  Created by Christopher Minar on 7/2/25.
//

import SwiftUI
import Charts
import Combine

struct WeeklyStat: Identifiable, Hashable {
    let id: String
    let items: Int
    let daysSinceJan1st2025: Int
    
    init(week: String, items: Int, daysSinceJan1st2025: Int) {
        self.id = week
        self.items = items
        self.daysSinceJan1st2025 = daysSinceJan1st2025
    }
}

@MainActor
class WeeklyPlotViewModel: ObservableObject {
    @Published var weeklyStats: [WeeklyStat] = []
    @Published var isLoading = true
    @Published var errorMessage: String?
    private var cancellables = Set<AnyCancellable>()
    var monthInts: [Int]

    init() {
        self.monthInts = []
        StateManager.shared.$user
            .compactMap { $0 } // Ensure user is not nil
            .first() // Take the first non-nil value
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in
                Task {
                    await self?.fetchWeeklyStats()
                }
            }
            .store(in: &cancellables)
    }

    func fetchWeeklyStats() async {
        guard let piId = StateManager.shared.user?.piIds?.first else {
            errorMessage = "No Recycla Pi paired."
            isLoading = false
            return
        }
        
        isLoading = true
        do {
            let stats = try await PiManager.shared.getWeeklyStats(piId: piId)
            weeklyStats = stats.map { WeeklyStat(week: $0.week, items: $0.items, daysSinceJan1st2025: $0.daysSinceJan1st2025) }
            
            if let firstStat = weeklyStats.first, let lastStat = weeklyStats.last {
                let calendar = Calendar.current
                let year2025 = calendar.date(from: DateComponents(year: 2025, month: 1, day: 1))!
                
                let firstDate = calendar.date(byAdding: .day, value: firstStat.daysSinceJan1st2025, to: year2025)!
                let lastDate = calendar.date(byAdding: .day, value: lastStat.daysSinceJan1st2025, to: year2025)!
                
                let firstMonth = calendar.component(.month, from: firstDate)
                let lastMonth = calendar.component(.month, from: lastDate)
                
                var monthDays: [Int] = []
                for month in firstMonth...lastMonth {
                    let firstOfMonthComponents = DateComponents(year: 2025, month: month, day: 1)
                    if let firstOfMonthDate = calendar.date(from: firstOfMonthComponents) {
                        let components = calendar.dateComponents([.day], from: year2025, to: firstOfMonthDate)
                        if let day = components.day {
                            monthDays.append(day)
                        }
                    }
                }
                self.monthInts = monthDays
            }
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }
}

struct WeeklyPlotView: View {
    @StateObject private var viewModel = WeeklyPlotViewModel()

    var body: some View {
        VStack {
            if viewModel.isLoading {
                ProgressView()
            } else if let errorMessage = viewModel.errorMessage {
                Text(errorMessage)
            } else {
                let minDays = viewModel.weeklyStats.map { $0.daysSinceJan1st2025 }.min() ?? 0
                let maxDays = viewModel.weeklyStats.map { $0.daysSinceJan1st2025}.max() ?? 0

                Chart(viewModel.weeklyStats) { stat in
                    AreaMark(
                        x: .value("Day", stat.daysSinceJan1st2025),
                        y: .value("Items Recycled", stat.items)
                    )
                    .interpolationMethod(.catmullRom)
                    .foregroundStyle(
                        LinearGradient(
                            gradient: Gradient(colors: [Color.blue.opacity(0.9), Color.blue.opacity(0)]),
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    PointMark(
                        x: .value("Day", stat.daysSinceJan1st2025),
                        y: .value("Items Recycled", stat.items)
                    )
                    .foregroundStyle(.black)
                    .symbolSize(25)
                }
                .frame(height: 200)
                .chartXAxis {
                    AxisMarks(values: viewModel.weeklyStats.map { $0.daysSinceJan1st2025 }) { value in
                        AxisValueLabel(orientation: .verticalReversed) {
                            if let day = value.as(Int.self) {
                                if let stat = viewModel.weeklyStats.first(where: { $0.daysSinceJan1st2025 == day }) {
                                    Text(stat.id)
                                }
                            }
                        }.offset(x: -10)
                    }
                    /// vertical month markers
                   AxisMarks(values: viewModel.monthInts) { value in
                       AxisGridLine()
                   }
                }
                .chartXScale(domain: minDays-2...maxDays+2)
                
                Text("Date")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .navigationTitle("Weekly Recycling")
    }
}

#Preview {
    WeeklyPlotView()
}
