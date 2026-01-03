//
//  SettingsView.swift
//  recycla
//
//  Created by Christopher Minar on 1/23/25.
//

import SwiftUI


struct SettingsView: View {
    
    @StateObject private var viewModel = SettingsViewModel()
    @StateObject private var stateManager = StateManager.shared
    @Binding var showSignInView: Bool
    @State private var showNicknameAlert = false
    @State private var newPiNickname = ""
    
    var body: some View {
        NavigationStack {
            List {
                Section {
                    Text("UserId:")
                        .font(.headline)
                    
                    if let user = stateManager.user {
                        Text("  \(user.id)")
                            .frame(maxWidth: .infinity, alignment: .leading)
                    } else {
                        ProgressView()
                    }

                    Text("Your Pis:")
                        .font(.headline)
                    
                    PiStatusView(piIds: stateManager.user?.piIds)
                    
                } header: {
                    Text("Information")
                        .font(.headline)
                }
                
                ArmedView()
                
                Button("Log out") {
                    Task {
                        do {
                            try viewModel.signOut()
                            showSignInView = true
                            // TODO also switch view to home
                        } catch{
                            // todo should do real error handle
                            print(error)
                        }
                    }
                }
                .font(.headline)
                .foregroundColor(.white)
                .frame(height: 55)
                .frame(maxWidth: .infinity)
                .background(Color.blue)
                .cornerRadius(10)

                Button("Change Pi Nickname") {
                    newPiNickname = PiManager.shared.yourPiNickname
                    showNicknameAlert = true
                }
                .font(.headline)
                .foregroundColor(.white)
                .frame(height: 55)
                .frame(maxWidth: .infinity)
                .background(Color.blue)
                .cornerRadius(10)
                .alert("Change Pi Nickname", isPresented: $showNicknameAlert) {
                    TextField("Current: \(PiManager.shared.yourPiNickname)", text: $newPiNickname)
                    Button("Cancel") {
                        showNicknameAlert = false
                    }
                    Button("Save") {
                        let piId = stateManager.user?.piIds?.first ?? "?"
                        Task {
                            do {
                                PiManager.shared.yourPiNickname = newPiNickname
                                try await PiManager.shared.changePiNickname(piId: piId)
                            } catch {
                                print("Error changing Pi nickname: \(error)")
                            }
                        }
                        showNicknameAlert = false
                    }
                } message: {
                    Text("Enter a new nickname for your Pi. Current nickname: \(PiManager.shared.yourPiNickname)")
                }
                
                NavigationLink{
                    PiView()
                } label: {
                    Text("Manage Pis")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(height: 55)
                        .frame(maxWidth: .infinity)
                        .background(Color.blue)
                        .cornerRadius(10)
                }
            
                
                NavigationLink{
                    ChangeLogView()
                } label: {
                    Text("Change Log")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(height: 55)
                        .frame(maxWidth: .infinity)
                        .background(Color.blue)
                        .cornerRadius(10)
                }
                
                if stateManager.authProviders.contains(.email){
                    emailSection
                }
            }
        }
        .navigationTitle("Settings")
    }
}

extension SettingsView {
    private var emailSection: some View{
        Section{
            Button("Reset Password") {
                Task {
                    do {
                        try await viewModel.resetPassword()
                        print("PASSWORD RESET")
                    } catch{
                        // todo should do real error handle
                        print(error)
                    }
                }
            }
        } header: {
            Text("Email Functions")
        }
    }
}


#Preview {
    SettingsView(showSignInView: .constant(false))
}
