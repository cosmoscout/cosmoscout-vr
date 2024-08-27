////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// The DifferenceImage2DNode has two input sockets accepting Image2D connections and a single
// Image2D output socket. The DifferenceImage2DComponent serves as a kind of factory. Whenever a new
// node is created, the builder() method is called. It is required that the class is called
// <NAME>Component.
class DifferenceImage2DComponent extends Rete.Component {
  constructor() {
    // This name must match the DifferenceImage2DNode::sName defined in DifferenceImage2DNode.cpp.
    super("DifferenceImage2D");

    // This specifies the submenu from which this node can be created in the node editor.
    this.category = "Operations";
  }

  // Called whenever a new node of this type needs to be constructed.
  builder(node) {

    // This node has two inputs from which the difference is calculated.
    // The first parameter is the name of the socket and must be unique amngst all sockets.
    // It is also used in the DifferenceImage2DComponent::process() to read the inputs of this node.
    // The second parameter is the display name of the socket on the node.
    // The last parameter references a socket type which has been registered with the node factory
    // before.
    let firstInput = new Rete.Input('first', "First", CosmoScout.socketTypes['Image2D']);
    node.addInput(firstInput);

    let secondInput = new Rete.Input('second', 'Second', CosmoScout.socketTypes['Image2D']);
    node.addInput(secondInput);

    // This node has a single output.
    // The first parameter is the name of the socket and must be unique amongst all sockets.
    // It is also used in the DifferenceImage2DComponent::process() to write the output of this
    // node. The second parameter is shown as name on the node. The last parameter references a
    // socket type which has been registered with the node factory before.
    let output = new Rete.Output('value', "Output", CosmoScout.socketTypes['Image2D']);
    node.addOutput(output);

    // This node has a widget to display error messages
    const statusDisplay = new StatusDisplay('Errors');
    node.addControl(statusDisplay);

    // This node has a listener for Messages to handle validation and error state
    node.onMessageFromCPP = (message) => { statusDisplay.setStatus(message); };

    return node;
  }
}

class StatusDisplay extends Rete.Control {
  constructor(key) {
    super(key);

    // This HTML code will be used whenever a node is created with this widget
    this.template = `
      <p class="status-text"></p>

      <style>
        p .status-text {
          display: none;
          color: var(--cs-color-text);
          font-family: 'Ubuntu Mono', monospace;
          width: 150px;
          padding: 10px 15px; !important
          margin: 10px; !important
        }
      </style>
    `;
  }
  // This is called by the node.onMessageFromCPP method whenever the content needs to be modified
  setStatus(value) {
    // Each node container gets the id "#node-<id>".
    // This way we can select elements inside the node using a selector.
    const elem = document.querySelector(`#node-${this.parent.id} .status-text`);

    console.log(value);
    if (value.status == 'OK') {
      elem.style['display'] = 'none';
    } else {
      elem.style['display'] = 'block'
      elem.style['color']   = 'red';

      elem.innerHTML = ''
      value.error.forEach((err) => {
        switch (err) {
          case 'DimensionMismatch':
            elem.innerHTML += 'Input dimensions do not match';
            break;
          case 'BoundsMismatch':
            elem.innerHTML += 'Input bounds do not match';
            break;
          case 'NumScalarsMismatch':
            elem.innerHTML += 'Input scalar sizes do not match';
            break;
          case 'PointsTypeMismatch':
            elem.innerHTML += 'Input scalar types do not match';
            break;
          default:
            elem.innerHTML += value.status;
            break;
        }
        elem.innerHTML += '<br>'
      })
    }
  }
}