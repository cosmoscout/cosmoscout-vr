////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// The NumberNode has a single output socket and a custom widget for entering a number. The
// custom widget is defined further below.
// The NumberComponent serves as a kind of factory. Whenever a new node is created, the
// builder() method is called. It is required that the class is called <NAME>Component.
class NumberComponent extends Rete.Component {
  constructor() {
    // This name must match the NumberNode::sName defined in NumberNode.cpp.
    super("Number");

    // This specifies the submenu from which this node can be created in the node editor.
    this.category = "Inputs";
  }

  // Called whenever a new node of this type needs to be constructed.
  builder(node) {

    // This node has a single output. The first parameter is the name of the socket and must be
    // unique amongst all sockets. It is also used in the NumberComponent::process() to write
    // the output of this node. The second parameter is shown as name on the node. The last
    // parameter references a socket type which has been registered with the node factory
    // before.
    let output = new Rete.Output('value', "Output", CosmoScout.socketTypes['Number Value']);
    node.addOutput(output);

    // Add the number input widget. The name parameter must be unique
    // amongst all controls of this node. The NumberControl class is defined further below.
    let control = new NumberControl('number');
    node.addControl(control);

    // Once the HTML element for this node has been created, the node.onInit() method will be
    // called. This is used here to initialize the input widget. The node.data object may
    // contain a number as returned by NumberNode::getData() which - if present - should be
    // preselected.
    node.onInit = (nodeDiv) => { control.init(nodeDiv, node.data); };

    return node;
  }
}

// This is the widget which is used for inserting the number.
class NumberControl extends Rete.Control {
  constructor(key) {
    super(key);

    // This HTML code will be used whenever a node is created with this widget.
    this.template = `
          <input class="number-input" type="text" value="0" />

          <style>
            .number-input {
              margin: 10px 15px !important;
              width: 150px !important;
            }
          </style>
        `;
  }

  // This is called by the node.onInit() above once the HTML element for the node has been
  // created. If present, the data object may contain a number as returned by
  // NumberNode::getData() which - if present - should be preselected.
  init(nodeDiv, data) {

    // Get our input element.
    const el = nodeDiv.querySelector("input");

    // Preselect a number if one was given.
    if (data.value) {
      el.value = data.value;
    }

    // Send an update to the node editor server whenever the user enters a new value.
    el.addEventListener(
        'input', e => { CosmoScout.sendMessageToCPP(parseFloat(e.target.value), this.parent.id); });

    // Stop propagation of pointer move events. Else we would drag the node if we tried to
    // select some text in the input field.
    el.addEventListener('pointermove', e => e.stopPropagation());
  }
}
