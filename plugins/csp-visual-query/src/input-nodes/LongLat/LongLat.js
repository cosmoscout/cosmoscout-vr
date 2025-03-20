////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// The Real has a single output socket and a custom widget for entering a number. The
// custom widget is defined further below.
// The NumberComponent serves as a kind of factory. Whenever a new node is created, the
// builder() method is called. It is required that the class is called <NAME>Component.
class LongLatComponent extends Rete.Component {
  constructor() {
    // This name must match the LongLat::sName defined in LongLat.cpp.
    super("LongLat");

    // This specifies the submenu from which this node can be created in the node editor.
    this.category = "Input";
  }

  // Called whenever a new node of this type needs to be constructed.
  builder(node) {

    // This node has a single output. The first parameter is the name of the socket and must be
    // unique amongst all sockets. It is also used in the NumberComponent::process() to write
    // the output of this node. The second parameter is shown as name on the node. The last
    // parameter references a socket type which has been registered with the node factory
    // before.
    let valueOutput = new Rete.Output('value', "Output", CosmoScout.socketTypes['RVec2']);
    node.addOutput(valueOutput);

    // Add the number input widgets. The name parameter must be unique
    // amongst all controls of this node. The ButtonControl class is defined further below.
    let button = new ButtonControl(
        'button', (id) => { CosmoScout.sendMessageToCPP({foo: "bar"}, id); }, "Select Location");
    node.addControl(button);

    // Once the HTML element for this node has been created, the node.onInit() method will be
    // called. This is used here to initialize the input widget. The node.data object may
    // contain a number as returned by LongLat::getData() which - if present - should be
    // preselected.
    node.onInit = (nodeDiv) => { button.init(nodeDiv, node.data); };

    return node;
  }
}

// This is a Rete control which displays a button.
class ButtonControl extends Rete.Control {
  /**
   * Constructs a new ButtonControl.
   *
   * @param {string}                 key              A unique identifier within the parent node.
   * @param {InputChangeCallback}    clickCallback A callback, that gets called, when the user
   *     enters a new value
   * @param {string | undefined}     label            A label to show next to the options.
   */
  constructor(key, clickCallback, label = undefined) {
    super(key);

    this.callback = clickCallback;

    this.id = crypto.randomUUID();

    this.template = `<div>
        <button id="longlat-button-${this.id}" class="btn glass py-2 px-3 m-3">${label}</button>
      </div>`;
  }
  /** This is called by the node.onInit() once the HTML element for the node has been created. */
  init(nodeDiv, data) {
    this.el = nodeDiv.querySelector(`#longlat-button-${this.id}`);
    this.el.addEventListener('click', e => { this.callback(this.parent.id); });
  }
}