//
//  recyclaApp.swift
//  recycla
//
//  Created by Christopher Minar on 1/22/25.
//


import SwiftUI
import Firebase
import FirebaseMessaging
import UserNotifications

@main
struct recyclaApp: App {
    
    @UIApplicationDelegateAdaptor(AppDelegate.self) var delegate
    
    init() {
        // Configure Firebase before the app starts
        FirebaseApp.configure()
    }
    
    var body: some Scene {
        WindowGroup {
            NavigationStack{
                RootView()
            }
        }
    }
}

class AppDelegate: NSObject, UIApplicationDelegate, UNUserNotificationCenterDelegate, MessagingDelegate {
    
    static weak var shared: AppDelegate?
    private var pendingFCMToken: String?
    
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey : Any]? = nil) -> Bool {
        AppDelegate.shared = self
        
        UNUserNotificationCenter.current().delegate = self
        
        let authOptions: UNAuthorizationOptions = [.alert, .badge, .sound]
        UNUserNotificationCenter.current().requestAuthorization(
            options: authOptions,
            completionHandler: { granted, error in
                DispatchQueue.main.async {
                    application.registerForRemoteNotifications()
                }
            }
        )
        
        Messaging.messaging().delegate = self
        
        return true
    }
    
    func application(_ application: UIApplication, didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data) {
        Messaging.messaging().apnsToken = deviceToken
    }
    
    func application(_ application: UIApplication, didFailToRegisterForRemoteNotificationsWithError error: Error) {
        print("Failed to register for remote notifications: \(error)")
    }
    
    func messaging(_ messaging: Messaging, didReceiveRegistrationToken fcmToken: String?) {
        // Store the FCM token for later use
        self.pendingFCMToken = fcmToken
        
        // Try to update FCM token in Firestore if user is logged in
        if let fcmToken = fcmToken, let user = StateManager.shared.user {
            Task {
                do {
                    try await UserManager.shared.updateUserFcmToken(id: user.id, fcmToken: fcmToken)
                    self.pendingFCMToken = nil // Clear pending token after successful update
                } catch {
                    print("Failed to update FCM token in Firestore: \(error)")
                }
            }
        }
    }
    
    // Called when app receives a notification while in foreground
    func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent notification: UNNotification, withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void) {
        let userInfo = notification.request.content.userInfo
        
        // Show notification even when app is in foreground
        completionHandler([.banner, .sound, .badge])
    }
    
    // Called when user taps on notification
    func userNotificationCenter(_ center: UNUserNotificationCenter, didReceive response: UNNotificationResponse, withCompletionHandler completionHandler: @escaping () -> Void) {
        let userInfo = response.notification.request.content.userInfo
        
        // Handle notification tap here
        // e.g., navigate to specific screen based on notification data
        
        completionHandler()
    }
    
    // Called when app receives notification while in background/killed
    func application(_ application: UIApplication, didReceiveRemoteNotification userInfo: [AnyHashable: Any],
                     fetchCompletionHandler completionHandler: @escaping (UIBackgroundFetchResult) -> Void) {
        
        // Handle background notification here
        // This is called for data-only notifications or when app is in background
        
        completionHandler(UIBackgroundFetchResult.newData)
    }
    
    // Function to update FCM token when user becomes available
    func updatePendingFCMToken() {
        if let fcmToken = pendingFCMToken, let user = StateManager.shared.user {
            Task {
                do {
                    try await UserManager.shared.updateUserFcmToken(id: user.id, fcmToken: fcmToken)
                    self.pendingFCMToken = nil // Clear pending token after successful update
                } catch {
                    print("Failed to update pending FCM token in Firestore: \(error)")
                }
            }
        }
    }
}
