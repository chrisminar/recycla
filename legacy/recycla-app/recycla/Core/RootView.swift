//
//  RootView.swift
//  recycla
//
//  Created by Christopher Minar on 1/23/25.
//

import SwiftUI

struct RootView: View {
    
    @State private var showSignInView: Bool = false
    
    var body: some View {
        ZStack{
            if !showSignInView {
                NavigationStack{
                    HomeView(showSignInView: $showSignInView)
                }
            }
        }
        .onAppear {
            // try to authenticate from local, if fails set value to nil
            let authUser = try? AuthenticationManager.shared.getAuthenticatedUser()
            self.showSignInView = authUser == nil
        }
        .fullScreenCover(isPresented: $showSignInView) {
            NavigationStack {
                AuthenticationView(showSignInView: $showSignInView)
            }
        }
    }
}

#Preview {
    RootView()
}
