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
    // This name must match the TimeNode::NAME defined above.
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
    // node. The DisplayControl class is defined further below.
    let control = new DisplayControl('display');
    node.addControl(control);

    // Whenever a message from C++ arrives, we set the input value accordingly. This message is
    // sent by the DisplayNode::process() method.
    node.onMessageFromCPP = (message) => { control.setValue(message.value); };

    return node;
  }
}

// This is the widget which is used for displaying the data.
class DisplayControl extends Rete.Control {
  constructor(key) {
    super(key);

    // This HTML code will be used whenever a node is created with this widget.
    this.template = `
          <p class="display-value">0</p>

          <style>
            p.display-value {
              font-family: 'Ubuntu Mono', monospace;
              border-radius: var(--cs-border-radius-medium);
              background: rgba(255, 255, 255, 0.1);
              width: 200px;
              padding: 5px 15px;
              margin: 10px;
              text-align: right;
              font-size: 1.1em;
            }
          </style>
        `;
  }

  // This is called by the node.onMessageFromCPP method above whenever a new value is sent in
  // from C++.
  setValue(val) {

    // Each node container gets the id "#node-<id>". This way we can select elements inside the
    // node using a selector. Here we select the p element with the class "display-value" as
    // defined by the template above.
    const el     = document.querySelector("#node-" + this.parent.id + " .display-value");
    el.innerHTML = val;
  }
}
