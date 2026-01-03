//
//  ProductsManager.swift
//  recycla
//
//  Created by Christopher Minar on 1/29/25.
//

import Foundation
import FirebaseFirestore
import FirebaseDatabase

final class PiManager: ObservableObject {
    
    static let shared = PiManager()
    private let db = Firestore.firestore()
    private let piCollection = Firestore.firestore().collection("pis")
    private var summaryListener: ListenerRegistration? = nil
    private var friendListeners: [String: ListenerRegistration] = [:]
    private var videoListener: ListenerRegistration? = nil
    var databaseRef: DatabaseReference!
    var lastSeen: Date? = nil
    var rawConnectionStatus: RawConnectionStatus? = nil
    @Published var connectionStatus: ConnectionStatus = ConnectionStatus.server
    @Published var armedStatus: ArmedStatus = ArmedStatus.disarmed
    @Published var yourPiNickname: String = ""
    
    private init() {
    }
    
    private let encoder: Firestore.Encoder = {
        let encoder = Firestore.Encoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        return encoder
    }()
    
    private let decoder: Firestore.Decoder = {
        let decoder = Firestore.Decoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }()
    
    private func piDocument(id piId: String) -> DocumentReference {
        piCollection.document(piId)
    }

    func getAllPiIds() async throws -> [String] {
        /// Sadly this will fetch the whole document (inefficient)
        let snapshot = try await piCollection.getDocuments()
        return snapshot.documents.map { $0.documentID }
    }
    
    func videoCollection() -> CollectionReference {
        db.collection("videos")
    }
    
    func getPi(piId: String) async throws -> PiModel {
        try await piDocument(id: piId).getDocument(as: PiModel.self)
    }

    func getNickname(piId: String) async throws -> String? {
        let document = try await piDocument(id: piId).getDocument()
        return document.data()?["nickname"] as? String
    }

    func changePiNickname(piId: String) async throws {
        // Update the nickname in the Firestore document
        let data: [String: Any] = [
            "nickname": yourPiNickname
        ]
        try await piDocument(id: piId).updateData(data)
    }
    
    func getAllPis() async throws -> [PiModel] {
        try await piCollection.getDocuments(as: PiModel.self)
    }
    
    func getUnclaimedPis() async throws -> [PiModel] {
        try await piCollection
            .whereField("users", in: [[], [""]]) // [] is the default array in python, [""] is the default array when added from console
            .order(by: "date_added")
            .getDocuments(as: PiModel.self)
    }
    
    func addPiToUser(piId: String) async throws {
        // get userid
        let uid = try AuthenticationManager.shared.getAuthenticatedUser().uid
        
        // try to add pi to user
        try await UserManager.shared.addUserPi(id: uid, piId: piId)
    }
    
    func addUserToPi(piId: String) async throws {
        let uid = try AuthenticationManager.shared.getAuthenticatedUser().uid
        
        let data: [String: Any] = [
            "users": FieldValue.arrayUnion([uid])
        ]
        try await piDocument(id: piId).updateData(data)
    }
    
    func removePiFromUser(piId: String) async throws {
        // get userid
        let uid = try AuthenticationManager.shared.getAuthenticatedUser().uid
        
        // try to remove pi
        try await UserManager.shared.removeUserPi(id: uid, piId: piId)
    }
    
    func removeUserFromPi(piId: String) async throws {
        let uid = try AuthenticationManager.shared.getAuthenticatedUser().uid
        
        let data: [String: Any] = [
            "users": FieldValue.arrayRemove([uid])
        ]
        try await piDocument(id: piId).updateData(data)
    }
    
    func monthlyStatsString(_ date: Date) -> String {
        let month = Calendar.current.component(.month, from: date)
        let year = Calendar.current.component(.year, from: date)
        return String(format: "%02d-%d", month, year)
    }

    func weeklyStatsString(_ date: Date) -> String {
        let weekOfYear = Calendar.current.component(.weekOfYear, from: date) - 1
        let year = Calendar.current.component(.year, from: date)
        return "\(year)-W\(weekOfYear)"
    }

    func summaryDocumentReference(for piId: String, period: SummaryPeriod, date: Date) -> DocumentReference {
        if period == .monthly {
            let mstring = monthlyStatsString(date)
            return piDocument(id: piId).collection("monthly_stats").document(mstring)
        } else if period == .weekly {
            let wstring = weeklyStatsString(date)
            return piDocument(id: piId).collection("weekly_stats").document(wstring)
        } else {
            return piDocument(id: piId)
        }
    }

    func addListenerForSummary(id piId: String, period: SummaryPeriod, completion: @escaping (
        _ numItems: Int,
        _ materialPctCorrect: Double,
        _ subMaterialPctCorrect: Double,
        _ materialNumClass: [String: Int],
        _ subMaterialNumClass: [String: Int],
        _ materialPctClassCorrect: [String: Double],
        _ subMaterialPctClassCorrect: [String: Double]
    ) -> Void) {

        let ref = summaryDocumentReference(for: piId, period: period, date: Date())
        let registration = ref.addSnapshotListener { querySnapshot, error in
            if let error = error {
                print("Error fetching document: \(error.localizedDescription)")
                return
            }
            guard let document = querySnapshot, document.exists else {
                print("No pi document or document does not exist")
                return
            }

            guard let piModel = try? document.data(as: PiModelNoID.self, decoder: self.decoder) else {
                print("Pi Document decode failed")
                return
            }
            
            let numItems = piModel.numItems ?? 0
            let materialPctCorrect = piModel.materialPctCorrect ?? 0.0
            let subMaterialPctCorrect = piModel.subMaterialPctCorrect ?? 0.0
            let materialNumClass = piModel.materialNumClass ?? [:]
            let subMaterialNumClass = piModel.subMaterialNumClass ?? [:]
            let materialPctClassCorrect = piModel.materialPctClassCorrect ?? [:]
            let subMaterialPctClassCorrect = piModel.subMaterialPctClassCorrect ?? [:]

            completion(
                numItems,
                materialPctCorrect,
                subMaterialPctCorrect,
                materialNumClass,
                subMaterialNumClass,
                materialPctClassCorrect,
                subMaterialPctClassCorrect
            )
        }
        self.summaryListener = registration
    }

    func addListenerForFriends(ids piIds: [String], period: SummaryPeriod, date: Date, completion: @escaping (
        _ nickname: String,
        _ numItems: Int,
        _ bounty: Int
    ) -> Void) async throws {
        for piId in piIds {
            let nickname = try await getNickname(piId: piId)
            let ref = PiManager.shared.summaryDocumentReference(for: piId, period: period, date: date)
            // Assign the listener after creation to avoid capturing self strongly in the closure
            let listener = ref.addSnapshotListener { [weak self] snapshot, error in
                if let error = error {
                    print("Error fetching monthly stats for leaderboard: \(error.localizedDescription)")
                    return
                }
                var numItems: Int = 0
                var bounty: Int = 0
                if let document = snapshot, document.exists {
                    numItems = document.data()? ["num_items"] as? Int ?? 0
                    bounty = document.data()? ["bounty"] as? Int ?? 0
                } else {
                    print("Monthly stats document does not exist for piId: \(piId)")
                }
                completion(nickname ?? "Unknown", numItems, bounty)
            }
            self.friendListeners[nickname ?? "Unknown"] = listener
        }
    }
    
    
    func removeListenerForSummary(statsType: StatsType) {
        switch statsType {
        case .summary:
            self.summaryListener?.remove()
        case .friends:
            for (_, friendListener) in self.friendListeners {
                friendListener.remove()
            }
        }
    }
    
    func addListenerForVideos(id piId: String, completion: @escaping (_ videos: [String] ) -> Void) {
        print("trying to add listener")
        self.videoListener = piDocument(id: piId).addSnapshotListener { querySnapshot, error in
            guard let document = querySnapshot else {
                print("No document")
                return
            }
            
            guard let piModel = try? document.data(as: PiModel.self, decoder: self.decoder) else{
                print("Document decode failed")
                return
            }
            
            //let videoIds = piModel.videos?.suffix(7) ?? []
            let videoIds = piModel.videos ?? []

            completion(videoIds)
        }
    }
    
    func removeListenerForVideo() {
        self.videoListener?.remove()
    }
    
    private func formatWeekString(_ weekString: String) -> String {
        let components = weekString.components(separatedBy: "-W")
        guard components.count == 2,
              let yearString = components.first,
              let weekStringComponent = components.last,
              let year = Int(yearString),
              let week = Int(weekStringComponent) else {
            return weekString
        }

        var dateComponents = DateComponents()
        dateComponents.yearForWeekOfYear = year
        dateComponents.weekOfYear = week
        dateComponents.weekday = 2 // Monday

        let calendar = Calendar(identifier: .iso8601)
        if let date = calendar.date(from: dateComponents) {
            let formatter = DateFormatter()
            formatter.dateFormat = "MM-dd"
            return formatter.string(from: date)
        }

        return weekString
    }
    
    func getWeeklyStats(piId: String) async throws -> [(week: String, items: Int, daysSinceJan1st2025: Int)] {
        let calendar = Calendar(identifier: .iso8601)
        var last12WeeksData: [(weekString: String, date: Date)] = []
        let startOfWeek = calendar.date(from: calendar.dateComponents([.yearForWeekOfYear, .weekOfYear], from: Date())) ?? Date()

        for i in 0..<12 {
            if let date = calendar.date(byAdding: .weekOfYear, value: -(12 - i), to: startOfWeek) {
                let year = calendar.component(.yearForWeekOfYear, from: date)
                let week = calendar.component(.weekOfYear, from: date)
                let weekString = String(format: "%d-W%02d", year, week)
                last12WeeksData.append((weekString: weekString, date: date))
            }
        }

        let last12Weeks = last12WeeksData.map { $0.weekString }
        let snapshot = try await piDocument(id: piId).collection("weekly_stats").whereField(FieldPath.documentID(), in: last12Weeks).getDocuments()
        
        let statsDict = snapshot.documents.reduce(into: [String: Int]()) { (dict, doc) in
            dict[doc.documentID] = doc.data()["num_items"] as? Int ?? 0
        }

        let jan1st2025Components = DateComponents(year: 2025, month: 1, day: 1)
        guard let jan1st2025 = calendar.date(from: jan1st2025Components) else {
            throw NSError(domain: "PiManagerError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Could not create date for Jan 1st, 2025"])
        }

        let finalStats = last12WeeksData.map { weekData -> (week: String, items: Int, daysSinceJan1st2025: Int) in
            let formattedWeek = formatWeekString(weekData.weekString)
            let items = statsDict[weekData.weekString] ?? 0
            let days = calendar.dateComponents([.day], from: jan1st2025, to: weekData.date).day ?? 0
            return (week: formattedWeek, items: items, daysSinceJan1st2025: days)
        }
        
        return finalStats
    }
    
    func updateArmedStatus(piId: String, status: ArmedStatus) async throws {
        let refPath = "/devices/\(piId)/armed_status"
        let ref = Database.database().reference(withPath: refPath)
        try await ref.setValue(status.rawValue)
    }
}


extension Query {
    func getDocuments<T>(as type: T.Type) async throws -> [T] where T : Decodable {
        let snapshot = try await self.getDocuments()
        return try snapshot.documents.map({document in
            try document.data(as: T.self)
        })
    }
}
