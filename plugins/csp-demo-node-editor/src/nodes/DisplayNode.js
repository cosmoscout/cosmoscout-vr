////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// The DisplayNode has a single input socket and a custom widget for displaying the current
// value. The custom widget is defined further below.
// The DisplayComponent serves as a kind of factory. Whenever a new node is created, the
// builder() method is called. It is required that the class is called <NAME>Component.
class DisplayComponent extends Rete.Component {
  constructor() {
    // This name must match the DisplayNode::sName defined in DisplayNode.cpp.
    super("Display");

    // This specifies the submenu from which this node can be created in the node editor.
    this.category = "Outputs";
  }

  // Called whenever a new node of this type needs to be constructed.
  builder(node) {

    // This node has a single input. The first parameter is the name of this input and must be
    // unique amongst all sockets. It is also used in the DisplayNode::process() to read the
    // input of this node. The second parameter is shown as name on the node. The last parameter
    // references a socket type which has been registered with the node factory before.
    let input = new Rete.Input('number', "Number", CosmoScout.socketTypes['Number Value']);
    node.addInput(input);

    // Add the number display. The name parameter must be unique amongst all controls of this
    // node. The TextDisplayControl class is defined in the controls folder.
    let control = new TextDisplayControl('display', '0');
    node.addControl(control);

    // Whenever a message from C++ arrives, we set the input value accordingly. This message is
    // sent by the DisplayNode::process() method.
    node.onMessageFromCPP = (message) => { control.setValue(message.value); };

    return node;
  }
}
