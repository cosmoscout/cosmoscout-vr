////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

/**
 * @callback SliderChangeCallback
 * @param {number} value
 */

/** This is the widget which is used for selecting an option from a drop down menu. */
class SliderControl extends Rete.Control {
  /**
   * Constructs a new SliderControl.
   *
   * @param {string}               key              A unique identifier within the parent node.
   * @param {SliderChangeCallback} onChangeCallback A callback that gets called when the user
   *                                                moves the slider.
   * @param {SliderChangeCallback} onReleaseCallback A callback that gets called when the user
   *                                                releases the slider.
   * @param {string | undefined}   label            A label to show next to the options.
   * @param {number | undefined}   width            The width of the control in pixels. Defaults
   *                                                to 220.
   */
  constructor(key, onChangeCallback, onReleaseCallback, label = undefined, width = 220) {
    super(key);

    this.changeCallback  = onChangeCallback;
    this.releaseCallback = onReleaseCallback;

    this.id = crypto.randomUUID();

    this.template = `<div id='slider-container-${this.id}' class='container-fluid'>`;

    if (label) {
      this.template += `<div class="row">${label}:</div>`;
    }

    this.template += `
      <div class="row">
        <div id="slider-${this.id}" class="slider-${this.key}"></div>
      </div>
    `;

    this.template += `
      </div>
      <style>
        #slider-container-${this.id} {
          margin: 10px 15px !important;
        }
        #slider-${this.id} {
          width: ${width}px;
          margin: 5px 0 10px 0;

          .noUi-base {
             margin: 0;
          }
        }
      </style>
    `;
  }

  /** This is called by the node.onInit() once the HTML element for the node has been created. */
  init(nodeDiv, min, max, step, value) {
    this.el = nodeDiv.querySelector(`#slider-${this.id}`);

    this.el.addEventListener("pointermove", (event) => { event.stopPropagation(); });
    this.el.addEventListener("pointerdown", (event) => { event.stopPropagation(); });
    this.el.addEventListener("pointerup", (event) => { event.stopPropagation(); });

    noUiSlider.create(this.el, {
      start: value,
      step: step,
      range: {min, max},
    });

    this.el.noUiSlider.on(
        "slide", (values, handle, unencoded) => { this.changeCallback(unencoded); });
    this.el.noUiSlider.on(
        "set", (values, handle, unencoded) => { this.releaseCallback(unencoded); });
  }
}
