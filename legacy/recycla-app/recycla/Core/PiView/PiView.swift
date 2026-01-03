//
//  PiView.swift
//  recycla
//
//  Created by Christopher Minar on 1/29/25.
//

import SwiftUI


enum PiNotFoundError: Error, LocalizedError {
    case itemNotFound

    var errorDescription: String? {
        switch self {
        case .itemNotFound:
            return NSLocalizedString("The item does not exist in the list.", comment: "")
        }
    }
}

@MainActor
final class PiViewModel: ObservableObject {
    
    @Published private(set) var pis: [PiModel] = []
    @Published private(set) var myPis: [PiModel] = []
    @Published var selectedFilter: FilterOption = FilterOption.unclaimed
    
    func getAllPis() async throws {
        self.pis = try await PiManager.shared.getAllPis()
    }
    
    enum FilterOption: String, CaseIterable {
        case all
        case unclaimed
    }
    
    func filterSelected(option: FilterOption) async throws {
        switch option {
        case .all:
            self.pis = try await PiManager.shared.getAllPis()
            break
        case .unclaimed:
            self.pis = try await PiManager.shared.getUnclaimedPis()
            break
        }
        
        self.selectedFilter = option
    }
    
    func addPi(pi: PiModel) async throws {
        guard let index = self.pis.firstIndex(where: {$0.id == pi.id}) else {
            throw PiNotFoundError.itemNotFound
        }
        //update database
        try await PiManager.shared.addPiToUser(piId: pi.id)
        try await PiManager.shared.addUserToPi(piId: pi.id)
        
        // update local
        self.myPis.append(pi)
        self.pis.remove(at: index)
        
        // update state manager user
        if var piIds = StateManager.shared.user?.piIds {
            piIds.append(pi.id)
            StateManager.shared.user?.piIds = piIds
        } else {
            StateManager.shared.user?.piIds = [pi.id]
        }

        // update listeners
        PiManager.shared.removeListenerForSummary(statsType: .friends)
        PiManager.shared.removeListenerForSummary(statsType: .summary)
        PiManager.shared.removeListenerForVideo()
        StateManager.shared.addListenerSummary(period: StateManager.shared.currentSummaryPeriod, statsType: .summary)
        StateManager.shared.addListenerSummary(period: StateManager.shared.currentFriendsPeriod, statsType: .friends, date: Date())
        StateManager.shared.addListenerForVideos()
    }

    func removePi(pi: PiModel) async throws {
        // find index in myPis
        guard let index = self.myPis.firstIndex(where: {$0.id == pi.id}) else {
            throw PiNotFoundError.itemNotFound
        }
        
        //update database
        try await PiManager.shared.removePiFromUser(piId: pi.id)
        try await PiManager.shared.removeUserFromPi(piId: pi.id)
        
        // update local
        self.pis.append(pi)
        self.myPis.remove(at: index)
        
        // update user
        if let userIndex = StateManager.shared.user?.piIds?.firstIndex(of: pi.id) {
            StateManager.shared.user?.piIds?.remove(at: userIndex)
        }
        
        // update listener
        PiManager.shared.removeListenerForSummary(statsType: .summary)
        PiManager.shared.removeListenerForSummary(statsType: .friends)
        PiManager.shared.removeListenerForVideo()
        StateManager.shared.reset()
    }
    
    func getMyPis() async throws {
        let uid = try AuthenticationManager.shared.getAuthenticatedUser().uid
        let currentUser = try await UserManager.shared.getUser(id: uid)
        self.myPis.removeAll()
        for myPi in currentUser.piIds ?? [] {
            self.myPis.append(PiModel(id: myPi, dateAdded: nil, videos: nil, users: nil))
        }
    }
    
    func addNewMyPi(pi: PiModel) {
        self.myPis.append(pi)
    }
}


struct PiView: View {
    
    @StateObject private var viewModel = PiViewModel()
    
    var body: some View {
        List {
            Section {
                ForEach(viewModel.myPis) { pi in
                    HStack {
                        Image(systemName: "chevron.left")
                            .foregroundColor(.gray)
                        Image(systemName: "chevron.left")
                            .foregroundColor(.gray)
                        Image(systemName: "chevron.left")
                            .foregroundColor(.gray)
                        Spacer()
                        Text(pi.id)
                    }
                    .swipeActions(edge: .trailing) {
                        Button("UnClaim Device") {
                            Task {
                                try await viewModel.removePi(pi: pi)
                            }
                        }
                        .tint(.red)
                        .labelStyle(.iconOnly)
                        .background(Color.red)
                    }
                }
            } header: {
                VStack(alignment: .leading) {
                    Text("Your Devices")
                        .font(.headline)
                    Text("Swipe to change")
                        .font(.subheadline)
                }
            }
            
            Section {
                ForEach(viewModel.pis) { pi in
                    HStack {
                        Text(pi.id)
                        Spacer()
                        Image(systemName: "chevron.right")
                            .foregroundColor(.gray)
                        Image(systemName: "chevron.right")
                            .foregroundColor(.gray)
                        Image(systemName: "chevron.right")
                            .foregroundColor(.gray)
                    }
                    .swipeActions(edge: .leading) {
                        Button("Claim Device") {
                            Task {
                                try await viewModel.addPi(pi: pi)
                            }
                        }
                        .tint(.green)
                        .labelStyle(.iconOnly)
                        .background(Color.green)
                    }
                }
//                .onDelete { index in
//                    Task {
//                        try await viewModel.addUserToPi(at: index.first!)
//                    }
//                }
            } header: {
                Text("Available Devices")
                    .font(.headline)
            }
        }
        .navigationTitle("Pi Manager")
        .toolbar(content: {
            ToolbarItem(placement: .automatic) {
                Menu("Filter: \(viewModel.selectedFilter.rawValue)") {
                    ForEach(PiViewModel.FilterOption.allCases, id: \.self) { filterOption in
                        Button(filterOption.rawValue) {
                            Task {
                                try? await viewModel.filterSelected(option: filterOption)
                            }
                        }
                    }
                }
            }
        })
        .task {
            try? await viewModel.filterSelected(option: viewModel.selectedFilter)
            try? await viewModel.getMyPis()
        }
    }
}

#Preview {
    //NavigationStack {
        PiView()
    //}
}
