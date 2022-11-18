////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// The TimeNode is pretty simple as it only has a single output socket. The component serves as
// a kind of factory. Whenever a new node is created, the builder() method is called.
class TimeComponent extends Rete.Component {

  constructor() {
    // This name must match the TimeNode::sName defined in TimeNode.cpp.
    super("Time");

    // This specifies the submenu from which this node can be created in the node editor.
    this.category = "Inputs";
  }

  // Called whenever a new node of this type needs to be constructed.
  builder(node) {

    // This node has a single output. The first parameter is the name of this output and must be
    // unique amongst all sockets. It is also used in the TimeNode::process() to write the
    // output of this node. The second parameter is shown as name on the node. The last
    // parameter references a socket type which has been registered with the node factory
    // before. It is required that the class is called <NAME>Component.
    let output = new Rete.Output('time', "Seconds", CosmoScout.socketTypes['Number Value']);
    node.addOutput(output)

    return node;
  }
}
