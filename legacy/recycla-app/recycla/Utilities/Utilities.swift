//
//  Utilities.swift
//  recycla
//
//  Created by Christopher Minar on 1/24/25.
//

import Foundation
import UIKit
import SwiftUI

struct OnFirstAppearViewModifier: ViewModifier {
    
    @State private var didAppear: Bool = false
    let perform: (() -> Void)?

    func body(content: Content) -> some View {
        content
            .onAppear {
                if !didAppear {
                    perform?()
                    didAppear = true
                }
            }
    }
}

extension View {
    
    func onFirstAppear(perform: (() -> Void)?) -> some View {
        modifier(OnFirstAppearViewModifier(perform: perform))
    }
    
}

enum StatsType {
    case summary
    case friends
}

final class Utilities {
    
    static let shared = Utilities()
    private init() {}
    
    @MainActor
    func topViewController(controller: UIViewController? = nil) -> UIViewController? {
        
        let controller = controller ?? UIApplication.shared.keyWindow?.rootViewController
        
        if let navigationController = controller as? UINavigationController {
            return topViewController(controller: navigationController.visibleViewController)
        }
        if let tabController = controller as? UITabBarController {
            if let selected = tabController.selectedViewController {
                return topViewController(controller: selected)
            }
        }
        if let presented = controller?.presentedViewController {
            return topViewController(controller: presented)
        }
        return controller
    }
    
    func timeSince(_ date: Date) -> String {
        let formatter = DateComponentsFormatter()
        formatter.unitsStyle = .abbreviated
        formatter.allowedUnits = [.year, .month, .day, .hour, .minute, .second]
        formatter.maximumUnitCount = 1 // Show only the largest unit

        let now = Date()
        let timeInterval = now.timeIntervalSince(date)
        
        // Had a thing where the time was displaying as negative
        // likely due to a bug on the pi where the date was reported wrong
        // This prevents negative time intervals
        if timeInterval < 0 {
            return "0s ago"
        }

        guard let formattedString = formatter.string(from: timeInterval) else {
            return "Just now"
        }

        return "\(formattedString) ago"
    }
}
