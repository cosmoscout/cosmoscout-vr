////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

/**
 * @callback InputChangeCallback
 */

/** This is the widget which is used to input a real number. */
class RealInputControl extends Rete.Control {
  /**
   * Constructs a new RealInputControl.
   *
   * @param {string}                 key              A unique identifier within the parent node.
   * @param {InputChangeCallback}    onChangeCallback A callback, that gets called, when the user enters a new value
   * @param {string | undefined}     label            A label to show next to the options.
   * @param {number}                 defaultValue     An initial value.
   */
  constructor(key, onChangeCallback, label = undefined, defaultValue = 0) {
    super(key);

    this.callback = onChangeCallback;

    this.id = crypto.randomUUID();

    this.template = `<div>`;
    if (label) {
      this.template += `
     <label for="real-input-${this.id}">${label}:</label>
     `;
    }

    this.template += `
        <input id="real-input-${this.id}" class="number-input" type="text" value="${defaultValue}"/>
      </div>
      <style>
        .number-input {
          margin: 10px 15px !important;
          width: 150px !important;
        }
      </style>
    `;

    let x = new RegExp(/([^\d.-])/g); // matches every character that is not a digit, dot or minus
    let y = new RegExp(/(?<=.)-/g); // matches any minus character that is not the first character
    let z = new RegExp(/(?<=(.*\..*))\./g); // matches any dot that comes after the first occurrence
    this.realRegex = new RegExp(x.source + "|" + y.source + "|" + z.source);
  }

  /** This is called by the node.onInit() once the HTML element for the node has been created. */
  init(nodeDiv, data) {
    // Initialize the bootstrap select.
    this.el = nodeDiv.querySelector(`#real-input-${this.id}`);

    if (data.value) {
      this.el.value = data.value;
    }

    this.el.addEventListener(
      'input', e => {
        e.target.value = e.target.value.replace(this.realRegex, "");
        this.callback({
          value: parseFloat(e.target.value),
          id: this.parent.id
        })
      }
    );

    // Stop propagation of pointer move events. Else we would drag the node if we tried to
    // select some text in the input field.
    this.el.addEventListener('pointermove', e => e.stopPropagation());
  }

  /**
   * TODO
   * @param {number} newValue new number to be entered
   */
  setValue(newValue) {

  }

  /**
   * TODO
   * @param {bool} status
   */
  enableInput(status) {

  }
}