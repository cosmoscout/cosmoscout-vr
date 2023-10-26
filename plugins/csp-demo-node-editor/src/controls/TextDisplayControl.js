////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// This is the widget which is used for displaying text data.
class TextDisplayControl extends Rete.Control {
  constructor(key, initialValue = '') {
    super(key);

    // This HTML code will be used whenever a node is created with this widget.
    this.template = `
          <p class="text-value">${initialValue}</p>

          <style>
            p.text-value {
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

  // This can be called by the node.onMessageFromCPP method whenever a new value is sent in from
  // C++.
  setValue(val) {

    // Each node container gets the id "#node-<id>". This way we can select elements inside the
    // node using a selector. Here we select the p element with the class "text-value" as
    // defined by the template above.
    const el     = document.querySelector("#node-" + this.parent.id + " .text-value");
    el.innerHTML = val;
  }
}
