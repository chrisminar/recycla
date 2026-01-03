//
//  ArmedView.swift
//  recycla
//
//  Created by Christopher Minar on 9/29/25.
//

import SwiftUI

struct ArmedView: View {
    @StateObject private var piManager = PiManager.shared
    @StateObject private var stateManager = StateManager.shared
    
    private var isArmed: Bool {
        piManager.armedStatus == .armed
    }
    
    var body: some View {
        VStack(spacing: 20) {
           
            HStack(spacing: 0) {
                Button(action: {
                    Task {
                        await updateArmedStatus(to: .armed)
                    }
                }) {
                    Text(isArmed ? "Armed" : "Arm")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(isArmed ? .white : .primary)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(isArmed ? Color.green : Color.clear)
                }
                
                Button(action: {
                    Task {
                        await updateArmedStatus(to: .disarmed)
                    }
                }) {
                    Text(!isArmed ? "Disarmed" : "Disarm")
                        .font(.system(size: 16, weight: .medium))
                        .foregroundColor(!isArmed ? .white : .primary)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(!isArmed ? Color.red : Color.clear)
                }
            }
            .background(Color.gray.opacity(0.2))
            .cornerRadius(8)
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(Color.gray.opacity(0.3), lineWidth: 1)
            )
        }
        .padding()
    }
    
    private func updateArmedStatus(to status: ArmedStatus) async {
        guard let piId = stateManager.user?.piIds?.first else {
            print("No Pi ID available")
            return
        }
        
        do {
            try await piManager.updateArmedStatus(piId: piId, status: status)
            print("Status: \(status.description)")
        } catch {
            print("Error updating armed status: \(error)")
        }
    }
}

#Preview {
    ArmedView()
}
