//
//  PiStatusManager.swift
//  recycla
//
//  Created by Christopher Minar on 3/18/25.
//

import Foundation
import FirebaseFirestore
import FirebaseDatabase
import SwiftUI


extension PiManager {
    func addPiStatusListener(id piId: String) {
        let refPath = "/devices/\(piId)"
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ss.SSSZZZZZ"
        databaseRef = Database.database().reference(withPath: refPath)

        getInitialStatusValues(dateFormatter: dateFormatter)
        
        addHealthListener(dateFormatter: dateFormatter)
        addConnectionStatusListener()
        addArmedStatusListener()
    }
    
    private func getInitialStatusValues(dateFormatter: DateFormatter) {
        let dispatchGroup = DispatchGroup()
        
        dispatchGroup.enter()
        databaseRef.observeSingleEvent(of: .value) { snapshot in
            if let value = snapshot.childSnapshot(forPath: "health").value as? String {
                self.lastSeen = dateFormatter.date(from: value)
            }
            dispatchGroup.leave()
        }
        
        dispatchGroup.enter()
        databaseRef.observeSingleEvent(of: .value) { snapshot in
            if let value = snapshot.childSnapshot(forPath: "connection_status").value as? String {
                self.rawConnectionStatus = RawConnectionStatus(rawValue: value)
            }
            dispatchGroup.leave()
        }
        
        dispatchGroup.enter()
        databaseRef.observeSingleEvent(of: .value) { snapshot in
            if let value = snapshot.childSnapshot(forPath: "armed_status").value as? String,
               let armedStatus = ArmedStatus(rawValue: value) {
                self.armedStatus = armedStatus
            } else {
                // No value exists, set default to armed
                self.armedStatus = .armed
                // Also set the value in the database
                self.databaseRef.child("armed_status").setValue(ArmedStatus.armed.rawValue)
            }
            dispatchGroup.leave()
        }
        
        dispatchGroup.notify(queue: .main) {
            self.updateConnectionStatus()
        }
    }
    
    private func getInitialConnectionStatusValue() {
        
    }
    
    private func addHealthListener(dateFormatter: DateFormatter) {
        databaseRef.observe(.childChanged) {
            snapshot in
            if snapshot.key == "health", let value = snapshot.value as? String {
                self.lastSeen = dateFormatter.date(from: value)
                self.updateConnectionStatus()
            }
        }
    }
    
    private func addConnectionStatusListener() {
        databaseRef.observe(.childChanged) {
            snapshot in
            if snapshot.key == "connection_status", let value = snapshot.value as? String {
                self.rawConnectionStatus = RawConnectionStatus(rawValue: value)
                self.updateConnectionStatus()
            }
        }
    }
    
    private func addArmedStatusListener() {
        databaseRef.observe(.childChanged) {
            snapshot in
            if snapshot.key == "armed_status",
               let value = snapshot.value as? String,
               let armedStatus = ArmedStatus(rawValue: value) {
                self.armedStatus = armedStatus
            }
        }
    }
    
    private func updateConnectionStatus() {
        self.connectionStatus = determineConnectionStatus(lastSeen: self.lastSeen, rawConnectionStatus: self.rawConnectionStatus)
    }
    
    private func determineConnectionStatus(lastSeen: Date?, rawConnectionStatus: RawConnectionStatus?) -> ConnectionStatus {
        if let lastSeen = lastSeen {
            if rawConnectionStatus == .disconnected || rawConnectionStatus == .offline {
                return ConnectionStatus.disconnected
            }
            else if Date().timeIntervalSince(lastSeen) < 60 {
                if (rawConnectionStatus == .connected) || (rawConnectionStatus == nil) {
                    return ConnectionStatus.ready
                } else if rawConnectionStatus == .processing {
                    return ConnectionStatus.processing
                } else { // stale
                    return ConnectionStatus.disconnected
                }
            } else {
                return ConnectionStatus.stale
            }
        } else { // last seen is nil when the value hasn't been read from database yet
            return ConnectionStatus.server
        }
    }
}


enum RawConnectionStatus: String {
    case connected = "connected"
    case disconnected = "disconnected"
    case offline = "offline"
    case processing = "processing"
}

enum ArmedStatus: String, CaseIterable {
    case armed = "armed"
    case disarmed = "disarmed"
    
    var color: Color {
        switch self {
        case .armed: return Color.green
        case .disarmed: return Color.red
        }
    }
    
    var description: String {
        switch self {
        case .armed: return "Device is armed"
        case .disarmed: return "Device is disarmed"
        }
    }
}

enum ConnectionStatus {
    case ready
    case processing
    case stale
    case disconnected
    case server
    
    var color: Color {
        switch self {
        case .ready: return Color.green
        case .processing: return Color.yellow
        case .stale: return Color.red
        case .disconnected: return Color.black
        case .server: return Color.blue
        }
    }
    
    var description: String {
        switch self {
        case .ready: return "Recycla ready."
        case .processing: return "Recycla processing."
        case .stale: return "Recycla has timed out."
        case .disconnected: return "Recycla disconnected."
        case .server: return "Waiting for database."
        }
    }
}
