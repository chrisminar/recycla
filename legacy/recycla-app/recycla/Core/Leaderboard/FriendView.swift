//
//  FriendView.swift
//  recycla
//
//  Created by Christopher Minar on 6/8/25.
//

import SwiftUI

@MainActor
final class FriendManager: BaseSummaryManager {
    static let shared = FriendManager()
    var piids: [String]
    @Published var date: Date
    @Published var dateRange: [Date]
    private override init() {
        self.piids = []
        self.date = Date()
        self.dateRange = []
        super.init()
        self.dateRange = getDateRange()
        self.date = dateRange[0]
        loadFilters()
    }
    
    func getPiIds() async throws {
        piids = try await PiManager.shared.getAllPiIds()
    }

    override func selectTimeFilter(_ period: SummaryPeriod) {
        if period == .all {
            self.timeFilterOption = .monthly
        } else if period == self.timeFilterOption {
            return
        } else {
            self.timeFilterOption = period
        }
        self.dateRange = getDateRange()
        self.date = self.dateRange[0]
        StateManager.shared.setSummaryPeriod(period, statsType: .friends, piIds: piids, date: date)
        saveFilters()
    }
    
    func dateString(_ newDate: Date) -> String {
        var dateString = ""
        switch timeFilterOption {
        case .monthly:
            dateString = monthString(for: newDate)
        case .weekly:
            dateString = weekIntervalString(for: newDate)
        default:
            dateString = ""
        }
        return dateString
    }

    func selectNewDate(_ newDate: Date) {
        self.date = newDate
        print("new date selected\(newDate), \(periodText(for: newDate))")
        StateManager.shared.setSummaryPeriod(self.timeFilterOption, statsType: .friends, piIds: piids, date: newDate)
    }
    
    func getDateRange() -> [Date] {
        switch timeFilterOption {
        case .weekly:
            // Get all weeks from this month and previous month (Monday to Sunday)
            var weeks: [Date] = []
            let calendar = Calendar.current
            let today = Date()
            // Start from the first day of previous month
            if let startDate = calendar.date(from: calendar.dateComponents([.year, .month], from: calendar.date(byAdding: .month, value: -1, to: today)!)) {
                var current = startDate
                let endDate = today
                while current <= endDate {
                    if calendar.component(.weekday, from: current) == 1 { // Sunday = 1
                        weeks.append(current)
                    }
                    current = calendar.date(byAdding: .day, value: 1, to: current)!
                }
            }
            if let lastWeekDay = weeks.last {
                self.date = lastWeekDay
            }
            return weeks.sorted(by: >)
        case .monthly:
            let calendar = Calendar.current
            let today = Date()
            if let oneMonthPrior = calendar.date(byAdding: .month, value: -1, to: today) {
                return [today, oneMonthPrior]
            } else {
                return [today]
            }
        default:
            return [Date()]
        }
    }

    override func saveFilters() {
        UserDefaults.standard.set(timeFilterOption.rawValue, forKey: "timeFilterOptionsFriends")
    }

    override func loadFilters() {
        if let savedPeriodRaw = UserDefaults.standard.string(forKey: "timeFilterOptionsFriends"),
           let savedPeriod = SummaryPeriod(rawValue: savedPeriodRaw) {
            selectTimeFilter(savedPeriod)
        } else {
            selectTimeFilter(.monthly)
        }
    }
}

struct FriendFilterButtonView: View {
    @StateObject var friendManager = FriendManager.shared
    
    var body: some View {
        Menu {
            ForEach(SummaryPeriod.allCases.filter { $0 != .all }, id: \.self) { period in
                Button {
                    friendManager.selectTimeFilter(period)
                } label: {
                    Label(period.rawValue.capitalized, systemImage: friendManager.timeFilterOption == period ? "checkmark.square" : "square")
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

struct DateButtonView: View {
    @StateObject var friendManager = FriendManager.shared
    
    var body: some View {
        Menu {
            ForEach(friendManager.dateRange, id: \.self) { date in
                Button {
                    friendManager.selectNewDate(date)
                } label: {
                    Label(friendManager.dateString(date), systemImage: friendManager.date == date ? "checkmark.square" : "square")
                }
            }
        } label: {
            Text(friendManager.periodText(for: friendManager.date))
                .font(.subheadline)
                .foregroundColor(.secondary)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(Color(.systemGray6))
                .cornerRadius(8)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(Color(.systemGray4), lineWidth: 0.5)
                )
        }
        .onChange(of: friendManager.dateRange.count) { oldCount, newCount in
            // This will trigger a UI update when dateRange changes
        }
        .onChange(of: friendManager.date) { oldDate, newDate in
            // This will trigger a UI update when the selected date changes
        }
    }
}

struct FriendView: View {
    @StateObject var friendManager = FriendManager.shared
    @State private var showBountyExplanation = false
    
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack {
                    HStack {
                        Spacer()
                        DateButtonView()
                        FriendFilterButtonView()
                    }
                    .padding(.bottom, 0)
                    
                    HStack {
                        Text("Name")
                            .font(.headline)
                            .foregroundColor(.primary)
                        Spacer()
                        Button {
                            showBountyExplanation = true
                        } label: {
                            HStack {
                                Text("Items Recycled\n(Bounty) Total")
                                    .font(.headline)
                                    .foregroundColor(.secondary)
                                Image(systemName: "questionmark.circle")
                                    .font(.headline)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .padding(.top, -6)
                }
                FriendListView()
            }
        }
        .padding()
        .navigationTitle("Friends")
        .alert("\(DateFormatter().monthSymbols[Calendar.current.component(.month, from: Date()) - 1]) Bounty", isPresented: $showBountyExplanation) {
            Button("OK") {
                showBountyExplanation = false
            }
        } message: {
            Text("\(DateFormatter().monthSymbols[Calendar.current.component(.month, from: Date()) - 1])'s bounty is for any recycled item with a material and sub material ground truth added.")
        }
    }
}

struct FriendListView: View {
    @StateObject var stateShared = StateManager.shared
    @StateObject var friendShared = FriendManager.shared
    @State private var sortedFriends: [(key: String, value: (Int, Int))] = []
    let datestr: String = {
        let formatter = DateFormatter()
        formatter.dateFormat = "LLLL"
        return formatter.string(from: Date())
    }()

    var body : some View {
        VStack {
            Divider()

            ForEach(Array(sortedFriends.enumerated()), id: \ .element.key) { (index, element) in
                FriendCellView(index: index, nickname: element.key, numitems: element.value.0, bounty: element.value.1)
            }
        }
        .onFirstAppear {
            Task {
                do {
                    try await friendShared.getPiIds()
                    // Use todays date for initialization
                    stateShared.addListenerSummary(period: stateShared.currentFriendsPeriod, statsType: .friends, piIds: friendShared.piids, date: Date())
                } catch {
                    print("Failed to load leaderboard: \(error)")
                }
            }
        }
        .onChange(of: stateShared.friendsStats.count) { oldCount, newCount in
            updateSortedFriends()
        }
        .onChange(of: stateShared.friendsStats.keys.sorted()) { oldKeys, newKeys in
            updateSortedFriends()
        }
        .onChange(of: stateShared.friendsStats.values.map { $0.0 }.reduce(0, +)) { oldSum, newSum in
            updateSortedFriends()
        }
        .onChange(of: stateShared.friendsStats.values.map { $0.1 }.reduce(0, +)) { oldSum, newSum in
            updateSortedFriends()
        }
        .onAppear {
            updateSortedFriends()
        }
    }
    
    private func updateSortedFriends() {
        sortedFriends = stateShared.friendsStats.sorted {
            if $0.value.1 != $1.value.1 {
                return $0.value.1 > $1.value.1 // sort by bounty
            } else if $0.value.1 != $1.value.1 {
                return $0.value.0 > $1.value.0 // then sort by total
            } else {
                return $0.key < $1.key // lastly sort by key
            }
        }
    }
}

struct FriendCellView: View {
    let index: Int
    let nickname: String
    let numitems: Int
    let bounty: Int
    var body : some View {
        HStack {
            Text(nickname)
                .font(.headline)
                .foregroundColor(.primary)
            // Crown for top 3
            if index == 0 && bounty != 0 {
                Image(systemName: "crown.fill")
                    .foregroundColor(.yellow)
            } else if index == 1 && bounty != 0 {
                Image(systemName: "crown.fill")
                    .foregroundColor(.gray)
            } else if index == 2 && bounty != 0 {
                Image(systemName: "crown.fill")
                    .foregroundColor(.orange)
            }
            Spacer()
            if numitems == 0 {
                Text("🙁")
                    .font(.headline)
            } else {
                Text("(\(bounty)) \(numitems)")
                    .font(.headline)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
    }
}

#Preview {
    FriendView()
}
