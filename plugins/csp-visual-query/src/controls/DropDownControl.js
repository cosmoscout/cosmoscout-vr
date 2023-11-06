////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

/**
 * @typedef {Object} DropDownOption
 * @property {number} value
 * @property {string} text
 */

/**
 * @callback DropDownChangeCallback
 * @param {DropDownOption}
 */

/** This is the widget which is used for selecting an option from a drop down menu. */
class DropDownControl extends Rete.Control {
  /**
   * Constructs a new DropDownControl.
   *
   * @param {string}                 key              A unique identifier within the parent node.
   * @param {DropDownChangeCallback} onChangeCallback A callback, that gets called, when the user
   *     selects a new option.
   * @param {string | undefined}     label            A label to show next to the options.
   * @param {Array<DropDownOption>}  defaultOptions   A set of initial options.
   */
  constructor(key, onChangeCallback, label = undefined, defaultOptions = []) {
    super(key);

    this.callback = onChangeCallback;

    let options = '';

    for (let option of defaultOptions) {
      options += `<option value="${option.value}">${option.text}</option>`
      options += '\n';
    }

    this.id = crypto.randomUUID();

    this.template = `<div id="dropdown-${this.id}" class="container-fluid">`;
    if (label) {
      this.template += `
        <div class="row">
          <label for="select-${this.id}">${label}:</label>
        </div>
      `;
    }

    this.template += `
      <div class="row">
        <select id="select-${this.id}" class="dropdown-${this.key}">
          ${options}
        </select>
      </div>
    `;
    this.template += `
      </div>
      <style>
       #dropdown-${this.id} {
          margin: 10px 15px !important;
          width: 150px !important;
        }
      </style>
    `;
  }

  /** This is called by the node.onInit() once the HTML element for the node has been created. */
  init(nodeDiv, data) {
    // Initialize the bootstrap select.
    this.el = nodeDiv.querySelector(`#select-${this.id}`);
    $(this.el).selectpicker();

    if (data.options) {
      this.setOptions(data.options);

      if (data.selectedValue) {
        $(this.el).selectpicker(
            'val', data.options.findIndex((entry) => entry.value === data.selectedValue ||
                                                     entry.text === data.selectedValue));
      }
    }
  }

  /**
   * Replaces the previous options with the new ones.
   * @param {Array<DropDownOption>} newOptions The new options to be shown.
   */
  setOptions(newOptions) {
    this.el.replaceChildren();

    for (let option of newOptions) {
      let optionElement       = document.createElement("option");
      optionElement.value     = option.value;
      optionElement.innerHTML = option.text;
      this.el.appendChild(optionElement);
    }

    // refresh bootstrap dropdown options
    $(this.el).selectpicker("refresh");

    // Send an update to the node whenever the user selects a new option.
    this.el.addEventListener('change', (e) => this.callback({
      value: parseInt(e.target.value),
      text: e.target.options[e.target.selectedIndex].text
    }));
  }
}
