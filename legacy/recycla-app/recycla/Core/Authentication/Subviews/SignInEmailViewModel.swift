//
//  SignInEmailViewModel.swift
//  recycla
//
//  Created by Christopher Minar on 1/28/25.
//

import Foundation

final class SignInEmailViewModel: ObservableObject {
    
    @Published var email = ""
    @Published var password = ""
    
    func signUp() async throws{
        guard !email.isEmpty, !password.isEmpty else{
            // todo should be real validation instead of print
            print("No email or password found.")
            return
        }
        let authDataResult = try await AuthenticationManager.shared.createUser(email: email, password: password)
        try await UserManager.shared.createNewUser(user: DBUser(auth: authDataResult))
    }
    
    func signIn() async throws{
        guard !email.isEmpty, !password.isEmpty else{
            // todo should be real validation instead of print
            print("No email or password found.")
            return
        }
        let authDataResult = try await AuthenticationManager.shared.signInUser(email: email, password: password)
        print("sign in with email success")
    }
}
