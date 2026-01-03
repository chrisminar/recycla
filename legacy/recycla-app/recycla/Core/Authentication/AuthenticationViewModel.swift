//
//  AuthenticationViewModel.swift
//  recycla
//
//  Created by Christopher Minar on 1/28/25.
//

import Foundation

@MainActor
final class AuthenticationViewModel: ObservableObject {
    
    func signInGoogle() async throws {
        let helper = SignInGoogleHelper()
        let tokens = try await helper.signIn()
        let authDataResult = try await AuthenticationManager.shared.signInWithGoogle(tokens: tokens)
        
        // check if uid exists in the database
        if await UserManager.shared.doesUserUserProfileExist(auth: authDataResult) {
            print("User already exists, do not make new profile")
        } else {
            let user = DBUser(auth: authDataResult)
            try await UserManager.shared.createNewUser(user: user)
        }
    }
}
