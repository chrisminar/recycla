//
//  HomeView.swift
//  recycla
//
//  Created by Christopher Minar on 2/6/25.
//

import SwiftUI


struct HomeView: View {
    
    @ObservedObject private var piManager = PiManager.shared
    @Binding var showSignInView: Bool
    @State private var showToast = false
    @State private var toastMessage = ""
    
    var body: some View {
        ZStack {
            TabView {
                RecycleSummaryView()
                    .tabItem {
                        Label("Summary", systemImage: "list.bullet")
                    }
                
                FriendView()
                    .tabItem {
                        Label("Friends", systemImage: "person.3.fill")
                    }
                
                ThumbnailView()
                    .tabItem {
                        Label("Thumbnail", systemImage: "photo")
                    }
                
                SettingsView(showSignInView: $showSignInView)
                    .tabItem {
                        Label("Settings", systemImage: "gearshape")
                    }
            }
            
            if showToast {
                ToastView(message: toastMessage, showToast: $showToast)
                    .transition(.move(edge: .top))
                    .onAppear {
                        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                            withAnimation {
                                showToast = false
                            }
                        }
                    }
            }
        }
        .onChange(of: piManager.connectionStatus) { _oldStatus, newStatus in
            toastMessage = newStatus.description
            withAnimation {
                showToast = true
            }
        }
    }
}

struct ToastView: View {
    let message: String
    @Binding var showToast: Bool
    
    var body: some View {
        VStack {
            Text(message)
                .padding()
                .frame(maxWidth: .infinity)
                .background(Color.black.opacity(0.8))
                .foregroundColor(.white)
                .cornerRadius(10)
                .padding(10)
                .padding(.top, 10)
                .gesture(
                    DragGesture()
                        .onEnded { value in
                            if value.translation.height < 0 {
                                withAnimation {
                                    showToast = false
                                }
                            }
                        }
                )
            Spacer()
        }
    }
}

#Preview {
    NavigationStack{
        HomeView(showSignInView: .constant(false))
    }
}
