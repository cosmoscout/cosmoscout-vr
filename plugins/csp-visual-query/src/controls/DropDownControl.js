////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// This is the widget which is used for selecting an option from a drop down menu.
class DropDownControl extends Rete.Control {
  constructor(key, defaultOptions = []) {
    super(key);

    let options = '';

    for (let option of defaultOptions) {
      options += `<option value="${option.value}">${option.text}</option>`
      options += '\n';
    }

    this.template = `
          <select>
            ${options}
          </select>

          <style>
            .dropdown {
              margin: 10px 15px !important;
              width: 150px !important;
            }
          </style>
        `;
  }

  // This is called by the node.onInit() above once the HTML element for the node has been
  // created.
  init(nodeDiv, data) {

    // Initialize the bootstrap select.
    const el = nodeDiv.querySelector("select");
    $(el).selectpicker();

    // Preselect a math operation.
    if (data.selectedValue) {
      $(el).selectpicker('val', data.selectedValue);
    }

    // Send an update to the node editor server whenever the user selects a new operation.
    el.addEventListener('change',
      (e) => { CosmoScout.sendMessageToCPP({
        value: parseInt(e.target.value),
        text: e.target.options[e.target.selectedIndex].text
      }, this.parent.id); });
  }

  setOptions(newOptions) {
    const el= document.querySelector("#node-" + this.parent.id + " select");
    el.replaceChildren();

    for (let option of newOptions) {
      let optionElement = document.createElement("option");
      optionElement.value = option.value;
      optionElement.innerHTML = option.text;
      el.appendChild(optionElement);
    }

    // refresh bootstrap dropdown options
    $(el).selectpicker("refresh");


    // Send an update to the node editor server whenever the user selects a new operation.
    el.addEventListener('change',
      (e) => { CosmoScout.sendMessageToCPP({
        value: parseInt(e.target.value),
        text: e.target.options[e.target.selectedIndex].text
      }, this.parent.id); });
  }
}
