import Foundation
import SwiftUI

@MainActor
class BaseSummaryManager: ObservableObject {
    @Published var timeFilterOption: SummaryPeriod = .monthly
    
    func selectTimeFilter(_ period: SummaryPeriod) {
        // To be overridden by subclasses
    }
    
    func saveFilters() {
        // To be overridden by subclasses
    }
    
    func loadFilters() {
        // To be overridden by subclasses
    }
    
    func periodText(for date: Date = Date()) -> String {
        switch timeFilterOption {
        case .all:
            return "Lifetime"
        case .monthly:
            return monthString(for: date)
        case .weekly:
            return weekIntervalString(for: date)
        }
    }
    
    func monthString(for date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "LLLL"
        return formatter.string(from: date)
    }

    func weekIntervalString(for date: Date) -> String {
        let calendar = Calendar.current
        let weekInterval = calendar.dateInterval(of: .weekOfYear, for: date)
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d"
        if let start = weekInterval?.start, let end = weekInterval?.end.addingTimeInterval(-1) {
            let startString = formatter.string(from: start)
            let endFormatter = DateFormatter()
            endFormatter.dateFormat = calendar.component(.month, from: start) == calendar.component(.month, from: end) ? "d" : "MMM d"
            let endString = endFormatter.string(from: end)
            return "\(startString) - \(endString)"
        }
        return ""
    }
}
