//
//  ChangeLogView.swift
//  recycla
//
//  Created by Christopher Minar on 3/3/25.
//

import SwiftUI

struct ChangeLogView: View {
    var changeLogEntries: [String] = [
        "1.5.0(1) - Add ability to arm/disarm from app",
        "1.4.9(1) - Label combination bugfix and update.",
        "1.4.8(1) - Display support for reading labels.",
        "1.4.7(1) - Thumbnails will now update as the cloud updates.",
        "1.4.6(2) - Users can now change pi nicknames.",
        "1.4.6(1) - Ability for user to see weekly/monthly and past weekly/monthly leaderboards. Added bounty to leaderboards.",
        "1.4.5(2) - Added push notifications.",
        "1.4.5(1) - Added plot.",
        "1.4.4(1) - Some sub materials will now be considered correct when chosen by the classifier. E.g. if the ground truth is 'plastic, plastic tub', 'plastic' will be considered correct.",
        "1.4.3(1) - Friends Tab Added.",
        "1.4.2(1) - Thumbnails can now be expanded by tapping the text area as well as the image.",
        "1.4.1(1) - Summary can now be sorted by week, month, alltime",
        "1.3.10(4) - App should now correctly handle thumbnails when a user updates category/brand without setting submaterial.",
        "1.3.10(3) - Thumbnails should no longer indefinitly pinwheel.",
        "1.3.10(2) - Fixed update ground truth in app reset when toggling image.",
        "1.3.10(1) - Can now scroll in summary view",
        "1.3.9(3) - Added default report options for dirty items and items with more than one material.",
        "1.3.9(2) - Updated recyclability strings to remind for cleanliness and lid removal.",
        "1.3.9(1) - Bugfix. If a thumbnail fails to fetch the image you will now still be able to click on pinwheel to expand the options and report it.",
        "1.3.8(1) - Added filtering menus to thumbnail view.",
        "1.3.7(1) - Remove product category",
        "1.3.6(1) - Add report button.",
        "1.3.5(1) - Not all items have the category/brand/product GT option now.",
        "1.3.4(1) - Combined submaterials and fixed related weirdness with display",
        "1.3.3(1) - Changing how thumbnail names are displayed.",
        "1.3.2(1) - Categroy ground truth now correctly loads if already set.",
        "1.3.1(2) - Various UI bug fixes.",
        "1.3.1(1) - Added ability to update category, brand, product ground truth.",
        "1.3.0(1) - Add hierarchical label support.",
        "1.2.1(1) - Fix pi status bug. Made summary scrollable. Sort Summary by most frequently recycled. Added taxonomy options.",
        "1.2.0(1) - rename to recycla",
        "1.1.5(1) - Added processing status to pi. Notifications when the device status changes.",
        "1.1.4(3) - Fixed crash when pi is added or removed? Negative 'time since' no longer possible.",
        "1.1.4(2) - Bug fixes. Status dot should not flicker anymore. Also shouldn't start red when the app boots and it should be green. Flickering of ground truth list should be fixed. list scroll flicker maybe fixed.",
        "1.1.4(1) - Setting an items groundtruth to background now removes it from your view",
        "1.1.3(1) - Thumbnail timers update. Thumbnail prediction color shouldn't break when image is selected",
        "1.1.2(1) - Fixed blank square in dark mode and dark mode shadows",
        "1.1.1(1) - Added changelog. Added a dot next to the pi in settings view. If its green it means the pi has responsed in the last 15seconds.",
        "1.0.0 - Initial release",
    ]
    
    var body: some View {
        NavigationStack {
            List(changeLogEntries, id: \.self) { entry in
                Text(entry)
            }
            .navigationTitle("Change Log")
        }
    }
}

#Preview {
    ChangeLogView()
}
