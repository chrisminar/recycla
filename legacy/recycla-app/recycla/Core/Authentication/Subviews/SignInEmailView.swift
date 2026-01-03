//
//  SignInEmailView.swift
//  recycla
//
//  Created by Christopher Minar on 1/23/25.
//

import SwiftUI

@MainActor
struct SignInEmailView: View {
    
    @StateObject private var viewModel = SignInEmailViewModel()
    @Binding var showSignInView: Bool
    
    var body: some View {
        VStack{
            TextField("Email...", text: $viewModel.email)
                .padding()
                .background(Color.gray.opacity(0.4))
                .cornerRadius(10)
            
            SecureField("Password...", text: $viewModel.password)
                .padding()
                .background(Color.gray.opacity(0.4))
                .cornerRadius(10)
            
            Button {
                Task{
                    // try to sign up
                    do {
                        try await viewModel.signUp()
                        showSignInView = false
                        return
                    } catch {
                        print("Attempting new account creation")
                        print(error)
                    }
                    
                    // try to sign in
                    do {
                        try await viewModel.signIn()
                        showSignInView = false
                        return
                    } catch{
                        print(error)
                    }
                }
                
            } label: {
                Text("Sign In")
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(height: 55)
                    .frame(maxWidth: .infinity)
                    .background(Color.blue)
                    .cornerRadius(10)
            }
            
            Spacer()
        }
        .padding()
        .navigationTitle("Sign In With Email")
    }
}


#Preview {
    NavigationStack{
        SignInEmailView(showSignInView: .constant(false))
    }
}
