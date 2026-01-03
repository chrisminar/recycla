//
//  UserManager.swift
//  recycla
//
//  Created by Christopher Minar on 1/28/25.
//

import Foundation
import FirebaseFirestore


final class UserManager {
    
    static let shared = UserManager()
    private let db = Firestore.firestore()
    
    private init() {
    }
    
    private func userDocument(id: String) -> DocumentReference {
        db.collection("users").document(id)
    }
    
    func videoCollection() -> CollectionReference {
        db.collection("videos")
    }
    
    private func userVideoCollection(id: String) -> CollectionReference {
        userDocument(id: id).collection("videos")
    }
    
    func doesUserUserProfileExist(auth: AuthDataResultModel) async -> Bool {
        let docRef = db.collection("users").document(auth.uid)
        do {
          let document = try await docRef.getDocument()
          if document.exists {
            print("User exists")
              return true
          } else {
            print("Document does not exist")
              return false
          }
        } catch {
          print("Error getting document: \(error)")
            return false
        }
    }
    
    let encoder: Firestore.Encoder = {
        let encoder = Firestore.Encoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        return encoder
    }()
    
    func createNewUser(user:DBUser) async throws {
        try userDocument(id: user.id).setData(from: user, merge: false, encoder: encoder)
    }
    
    func getUser(id: String) async throws -> DBUser {
        let documentSnapshot = try await userDocument(id: id).getDocument()
        guard documentSnapshot.exists else {
            throw NSError(domain: "UserManager", code: 404, userInfo: [NSLocalizedDescriptionKey: "User not found"])
        }
        let decoder = Firestore.Decoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try documentSnapshot.data(as: DBUser.self, decoder: decoder)
    }
    
    func addUserPi(id: String, piId: String) async throws {
        let data: [String: Any] = [
            "pi_ids": FieldValue.arrayUnion([piId])
        ]
        try await userDocument(id: id).updateData(data)
    }
    
    func removeUserPi(id: String, piId: String) async throws {
        let data: [String: Any] = [
            "pi_ids": FieldValue.arrayRemove([piId])
        ]
        try await userDocument(id: id).updateData(data)
    }
    
    func updateUserFcmToken(id: String, fcmToken: String) async throws {
        try await userDocument(id: id).updateData(["fcm_token": fcmToken])
    }
}
