//
//  UserModel.swift
//  recycla
//
//  Created by Christopher Minar on 2/13/25.
//

import Foundation

struct DBUser : Codable {
    let id: String
    let dateCreated: Date?
    let email: String?
    let photoUrl: String?
    var piIds: [String]?
    let videos: [String]?
    
    init(auth: AuthDataResultModel) {
        self.id = auth.uid
        self.dateCreated = Date()
        self.email = auth.email
        self.photoUrl = auth.photoUrl
        self.piIds = nil
        self.videos = nil
    }
    
    init (
        id: String,
        dateCreated: Date? = nil,
        email: String? = nil,
        photoUrl: String? = nil,
        piIds: [String]? = nil,
        videos: [String]? = nil
    ) {
        self.id = id
        self.dateCreated = dateCreated
        self.email = email
        self.photoUrl = photoUrl
        self.piIds = piIds
        self.videos = videos
    }
}
