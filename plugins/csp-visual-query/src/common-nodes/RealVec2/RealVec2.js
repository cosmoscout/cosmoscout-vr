////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// The Real has a single output socket and a custom widget for entering a number. The
// custom widget is defined further below.
// The NumberComponent serves as a kind of factory. Whenever a new node is created, the
// builder() method is called. It is required that the class is called <NAME>Component.
class RealVec2Component extends Rete.Component {
  constructor() {
    // This name must match the RealVec2::sName defined in RealVec2.cpp.
    super("RealVec2");

    // This specifies the submenu from which this node can be created in the node editor.
    this.category = "Constants";
  }

  // Called whenever a new node of this type needs to be constructed.
  builder(node) {

    // This node has a single output. The first parameter is the name of the socket and must be
    // unique amongst all sockets. It is also used in the NumberComponent::process() to write
    // the output of this node. The second parameter is shown as name on the node. The last
    // parameter references a socket type which has been registered with the node factory
    // before.
    let output = new Rete.Output('value', "Output", CosmoScout.socketTypes['RVec2']);
    node.addOutput(output);

    // Add the number input widgets. The name parameter must be unique
    // amongst all controls of this node. The NumberControl class is defined further below.
    let firstReal = new RealInputControl('firstReal', (callbackReturn) => {
      CosmoScout.sendMessageToCPP({first: callbackReturn.value}, callbackReturn.id);
    }, "", 0);
    node.addControl(firstReal);

    let secondReal = new RealInputControl('secondReal', (callbackReturn) => {
      CosmoScout.sendMessageToCPP({second: callbackReturn.value}, callbackReturn.id);
    }, "", 0);
    node.addControl(secondReal);

    // TODO: Add two input sockets to control the values of the vec by another real node

    // Once the HTML element for this node has been created, the node.onInit() method will be
    // called. This is used here to initialize the input widget. The node.data object may
    // contain a number as returned by RealVec2::getData() which - if present - should be
    // preselected.
    node.onInit = (nodeDiv) => { 
      firstReal.init(nodeDiv, node.data); 
      secondReal.init(nodeDiv, node.data); 
    };

    return node;
  }
}