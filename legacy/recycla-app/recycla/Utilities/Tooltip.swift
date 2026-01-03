//
//  Tooltip.swift
//  recyclo-app
//
//  Created by Christopher Minar on 3/7/25.
//

//import SwiftUI
//
//@available(OSX 10.15, *)
//public extension View {
//    func tooltip(_ toolTip: String) -> some View {
//        self.overlay(TooltipView(toolTip))
//    }
//}
//
//@available(OSX 10.15, *)
//private struct TooltipView: NSViewRepresentable {
//    let toolTip: String
//    
//    init(_ toolTip: String) {
//        self.toolTip = toolTip
//    }
//    
//    func makeNSView(context: NSViewRepresentableContext<TooltipView>) -> NSView {
//        NSView()
//    }
//    
//    func updateNSView(_ nsView: NSView, context: NSViewRepresentableContext<TooltipView>) {
//        nsView.toolTip = self.toolTip
//    }
//}
