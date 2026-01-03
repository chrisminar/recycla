//
//  StateManager.swift
//  recycla
//
//  Created by Christopher Minar on 2/14/25.
//

/// In perhaps what is the most ill advised programming decision of all time, I've made a singlton class here that all the views have access to to manage important state variables.
/// Not all of the important variables are in here, but I expect to slowly move them in over time as things get more complicated

import Foundation
import FirebaseMessaging
import UIKit

enum SummaryPeriod: String, CaseIterable, Identifiable {
    case all
    case monthly
    case weekly

    var id: String { self.rawValue }
}

struct SummaryStats {
    var numItems: Int = 0
    var materialPctCorrect: Double = 0.0
    var subMaterialPctCorrect: Double = 0.0
    var materialNumClass: [String: Int] = [:]
    var subMaterialNumClass: [String: Int] = [:]
    var materialPctClassCorrect: [String: Double] = [:]
    var subMaterialPctClassCorrect: [String: Double] = [:]
}

@MainActor
final class StateManager: ObservableObject {
    static let shared = StateManager()
    @Published var currentSummaryPeriod: SummaryPeriod = .all
    @Published var currentFriendsPeriod: SummaryPeriod = .all
    @Published var authProviders: [AuthProviderOption]
    @Published var user: DBUser?
    @Published private(set) var userVideos: [String]

    @Published var summaryStats: SummaryStats
    @Published var friendsStats: [String: (Int, Int)] = [:]

    init(
        summaryStats: SummaryStats = SummaryStats(),
        friendsStats: [String: (Int, Int)] = [:]
    ) {
        self.currentSummaryPeriod = .monthly
        self.currentFriendsPeriod = .monthly
        self.authProviders = []
        self.user = nil
        self.userVideos = []
        self.summaryStats = summaryStats
        self.friendsStats = friendsStats

        self.loadAuthProviders() // update authProviders

        Task{
            // load current user from firestore
            await self.loadcurrentUser()
            // if we got a user, add a status listener to the realtime database
            var piNickname: String = ""
            if let user = self.user, let piIds = user.piIds, !piIds.isEmpty {
                piNickname = try await PiManager.shared.getNickname(piId: piIds[0]) ?? piIds[0] // get pi nickname
                PiManager.shared.addPiStatusListener(id: piIds[0])
            }
            PiManager.shared.yourPiNickname = piNickname // save
        }
    }

    func reset() {
        // reset state to default values
        userVideos = []
        summaryStats = SummaryStats()
        friendsStats = [:]
    }
}

// user and pi
extension StateManager {
    func loadAuthProviders() {
        if let providers = try? AuthenticationManager.shared.getProviders() {
            authProviders = providers
        }
    }
    
    func loadcurrentUser() async {
        do {
            let authDataResult = try AuthenticationManager.shared.getAuthenticatedUser()
            let user = try await UserManager.shared.getUser(id: authDataResult.uid)
            await MainActor.run {
                self.user = user
            }
            
            // Update pending FCM token now that user is loaded
            DispatchQueue.main.async {
                if let appDelegate = AppDelegate.shared {
                    print("Attempting to update pending FCM token")
                    appDelegate.updatePendingFCMToken()
                } else {
                    print("Failed to get AppDelegate reference")
                }
            }
        } catch {
            print("Failed to load current user: \(error.localizedDescription)")
            // Handle error appropriately, e.g., show an alert to the user
        }
    }
}

// Summary functions
extension StateManager {
    func addListenerSummary(period: SummaryPeriod, statsType: StatsType, piIds: [String]? = nil, date: Date = Date()) {
        guard let authDataResult = try?
                AuthenticationManager.shared.getAuthenticatedUser() else { return }

        Task {
            // get user
            guard let user = try? await UserManager.shared.getUser(id: authDataResult.uid) else {
                print("failed to get user")
                //TODO better error handle
                return
            }

            //add listener
            do {
                switch statsType{
                case .summary:
                    // TODO, eventually this should handle more than one pi
                    guard let firstPiId = user.piIds?.first, !firstPiId.isEmpty else {
                        throw NSError(domain: "NoPiIdError", code: 0, userInfo: [NSLocalizedDescriptionKey: "No Pi ID found for the user."])
                    }
                    PiManager.shared.addListenerForSummary(id: firstPiId, period: period) { [weak self] numItems_, materialPctCorrect_, subMaterialPctCorrect_, materialNumClass_, subMaterialNumClass_, materialPctClassCorrect_, subMaterialPctClassCorrect_ in
                        self?.summaryStats.numItems = numItems_
                        self?.summaryStats.materialPctCorrect = materialPctCorrect_
                        self?.summaryStats.subMaterialPctCorrect = subMaterialPctCorrect_
                        self?.summaryStats.materialNumClass = materialNumClass_
                        self?.summaryStats.subMaterialNumClass = subMaterialNumClass_
                        self?.summaryStats.materialPctClassCorrect = materialPctClassCorrect_
                        self?.summaryStats.subMaterialPctClassCorrect = subMaterialPctClassCorrect_
                    }
                case .friends:
                    guard let piIds = piIds else {return}
                    try await PiManager.shared.addListenerForFriends(ids: piIds, period: period, date: date) { [weak self] nickname, numItems, bounty in
                        // Update the friendsStats dictionary with the nickname and numItems
                        self?.friendsStats[nickname] = (numItems, bounty)
                    }
                }
            } catch {
                print("Unable to add listener for summary, as this user has no pis associated with them.")
            }
        }
    }

    func setSummaryPeriod(_ period: SummaryPeriod, statsType: StatsType, piIds: [String]?=nil, date: Date = Date()) {
        /// This needs to be in a Task because it is publishing changes which isn't alowed from a view update
        Task { @MainActor in
            switch statsType {
            case .summary:
                currentSummaryPeriod = period
            case .friends:
                currentFriendsPeriod = period
            }
            /// Remove old listener
            PiManager.shared.removeListenerForSummary(statsType: statsType)
            /// Add new listener
            addListenerSummary(period: period, statsType: statsType, piIds: piIds, date: date)
        }
    }
}

// Thumbnail functions
extension StateManager {
    func addListenerForVideos() {
        guard let authDataResult = try? AuthenticationManager.shared.getAuthenticatedUser() else { return }
        
        Task {
            guard let user = try? await UserManager.shared.getUser(id: authDataResult.uid) else {
                print("failed to get user")
                //todo better error handle
                return
            }
            
            do {
                guard let firstPiId = user.piIds?.first, !firstPiId.isEmpty else {
                    // If user.piids is nil or empty
                    throw NSError(domain: "NoPiIdError", code: 0, userInfo: [NSLocalizedDescriptionKey: "No Pi ID found for the user."])
                }
                PiManager.shared.addListenerForVideos(id: firstPiId) { [weak self] videoIds in
                    self?.userVideos=videoIds
                }
            } catch {
                print("Unable to add listener for videos, as this user has no pis associated with them.")
            }
        }
    }
}

// add or remove pis
extension StateManager {
    func addUserPi(piId: String) {
        guard let user else { return }
        
        Task {
            do {
                try await UserManager.shared.addUserPi(id: user.id, piId: piId)
                let updatedUser = try await UserManager.shared.getUser(id: user.id)
                await MainActor.run {
                    self.user = updatedUser
                }
            } catch {
                print("Failed to add user Pi: \(error.localizedDescription)")
                // Handle error appropriately, e.g., show an alert to the user
            }
        }
    }
    
    func removeUserPi(piId: String) {
        guard let user else { return }
        
        Task {
            do {
                try await UserManager.shared.removeUserPi(id: user.id, piId: piId)
                let updatedUser = try await UserManager.shared.getUser(id: user.id)
                await MainActor.run {
                    self.user = updatedUser
                }
            } catch {
                print("Failed to remove user Pi: \(error.localizedDescription)")
                // Handle error appropriately, e.g., show an alert to the user
            }
        }
    }
}
