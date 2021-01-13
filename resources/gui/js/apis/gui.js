/* eslint no-param-reassign: 0 */

/**
 * This is a default CosmoScout API. Once initialized, you can access its methods via
 * CosmoScout.gui.<method name>.
 */
class GuiApi extends IApi {
  /**
   * @inheritDoc
   */
  name = 'gui';

  /**
   * Cache loaded templates
   *
   * @see {loadTemplateContent}
   * @type {Map<string, DocumentFragment>}
   * @private
   */
  _templates = new Map();

  /**
   * Registered html parts
   *
   * @see {registerHtml}
   * @type {Map<string, DocumentFragment>}
   * @private
   */
  _html = new Map();

  /**
   * Initialize third party drop downs,
   * add input event listener,
   * initialize tooltips
   */
  initInputs() {
    this.initDropDowns();
    this.initChecklabelInputs();
    this.initRadiolabelInputs();
    this.initTooltips();
    this.initPopovers();
    this.initDraggableWindows();
    this.initColorPickers();
  }

  /**
   * Initializes all .simple-value-dropdown with bootstrap's selectpicker.
   *
   * @see {initInputs}
   */
  initDropDowns() {
    const dropdowns = $('select.simple-value-dropdown:not(div.dropdown > select)');
    dropdowns.selectpicker();

    const eventListener = (event) => {
      if (event.target !== null) {
        let callback = CosmoScout.callbacks.find(event.target.dataset.callback)

        if (callback !== undefined) {
          callback(event.target.value);
        }
      }
    };

    document.querySelectorAll('select.simple-value-dropdown').forEach((dropdown) => {
      if (typeof dropdown.dataset.initialized === 'undefined') {
        dropdown.addEventListener('change', eventListener);

        dropdown.dataset.initialized = 'true';
      }
    });
  }

  /**
   * Adds a 'change' event listener which calls callNative with id and checkstate.
   * This will only add a listener once.
   *
   * @see {callNative}
   * @see {initInputs}
   */
  initChecklabelInputs() {
    document.querySelectorAll('.checklabel input').forEach((input) => {
      if (typeof input.dataset.initialized === 'undefined') {
        input.addEventListener('change', (event) => {
          if (event.target !== null) {
            let callback = CosmoScout.callbacks.find(event.target.dataset.callback)

            if (callback !== undefined) {
              callback(event.target.checked);
            }
          }
        });

        input.dataset.initialized = 'true';
      }
    });
  }

  /**
   * Adds a change event listener which calls callNative with the target id.
   *
   * @see {callNative}
   * @see {initInputs}
   */
  initRadiolabelInputs() {
    document.querySelectorAll('.radiolabel input').forEach((input) => {
      if (typeof input.dataset.initialized === 'undefined') {
        input.addEventListener('change', (event) => {
          if (event.target !== null) {
            let callback = CosmoScout.callbacks.find(event.target.dataset.callback)

            if (callback !== undefined) {
              callback(event.target.checked);
            }
          }
        });

        input.dataset.initialized = 'true';
      }
    });
  }

  /**
   * Initializes [data-toggle="tooltip"] elements.
   *
   * @see {initInputs}
   */
  initTooltips() {
    /* Bootstrap Tooltips require jQuery for now */
    $('[data-toggle="tooltip"]').tooltip({delay: 500});
  }

  /**
   * Initializes [data-toggle="popover"] elements.
   *
   * @see {initInputs}
   */
  initPopovers() {
    /* Bootstrap Popovers require jQuery for now */
    $('[data-toggle="popover"]').popover({
      html: true,
      content: function() {
        var content = this.getAttribute("data-popover-content");
        return document.querySelector(content).querySelector(".data-popover-body").innerHTML;
      },
      title: function() {
        var title = this.getAttribute("data-popover-content");
        return document.querySelector(title).querySelector(".data-popover-header").innerHTML;
      }
    })
  }

  initColorPickers() {
    const pickerDivs = document.querySelectorAll(".color-input");

    pickerDivs.forEach((pickerDiv) => {
      if (!pickerDiv.picker) {
        pickerDiv.picker = new CP(pickerDiv);
        pickerDiv.picker.self.classList.add('no-alpha');
        pickerDiv.picker.on('change', (r, g, b, a) => {
          const color                = CP.HEX([r, g, b, 1]);
          pickerDiv.style.background = color;
          pickerDiv.value            = color;
        });

        pickerDiv.oninput = (e) => {
          const color = CP.HEX(e.target.value);
          pickerDiv.picker.set(color[0], color[1], color[2], 1);
          pickerDiv.style.background = CP.HEX([color[0], color[1], color[2], 1]);
        };
      }
    });
  }

  /**
   * Initializes [class="draggable-window"] elements.
   *
   * @see {initInputs}
   */
  initDraggableWindows() {
    const windows     = document.querySelectorAll(".draggable-window");
    var currentZIndex = 100;

    windows.forEach((w) => {
      // Center initially.
      w.style.left = (document.body.offsetWidth - w.offsetWidth) / 2 + "px";
      w.style.top  = (document.body.offsetHeight - w.offsetHeight) / 2 + "px";

      // Make closable.
      const closeButton = w.querySelector(".window-header a[data-action='close']");
      if (closeButton) {
        closeButton.onmouseup = () => {
          w.classList.remove("visible");
        };
      }

      // Make lockable. Locked windows shall not automatically close.
      const lockButton = w.querySelector(".window-header a[data-action='lock']");
      if (lockButton) {
        w.locked             = false;
        lockButton.onmouseup = () => {
          w.locked = !w.locked;
          if (w.locked) {
            lockButton.querySelector("i").innerText = "lock";
          } else {
            lockButton.querySelector("i").innerText = "lock_open";
          }
        };
      }

      // Bring to front on click.
      w.onmousedown = () => {
        w.style.zIndex = ++currentZIndex;
      };

      // Make draggable.
      const header       = w.querySelector(".window-title");
      header.onmousedown = (e) => {
        w.startDragX = e.clientX;
        w.startDragY = e.clientY;

        document.onmouseup = () => {
          document.onmouseup   = null;
          document.onmousemove = null;
        };

        document.onmousemove = (e) => {
          e.preventDefault();

          // Do not move outside CosmoScout's window and leave some margin at the top and the bottom
          // of the screen so that we do not loose the window.
          const newTop = w.offsetTop + e.clientY - w.startDragY;

          if (e.clientX >= 0 && e.clientX < document.body.offsetWidth && newTop >= 20 &&
              newTop + 50 < document.body.offsetHeight) {
            w.style.left = (w.offsetLeft + e.clientX - w.startDragX) + "px";
            w.style.top  = newTop + "px";
            w.startDragX = e.clientX;
            w.startDragY = e.clientY;
          }
        };
      };
    });
  }

  /**
   * Appends a link stylesheet to the head.
   *
   * @param url {string}
   */
  registerCss(url) {
    const link = document.createElement('link');
    link.setAttribute('rel', 'stylesheet');
    link.setAttribute('href', url);

    document.head.appendChild(link);
  }

  /**
   * Removes a stylesheet by url.
   *
   * @param url {string}
   */
  unregisterCss(url) {
    document.querySelectorAll('link').forEach((element) => {
      if (typeof element.href !== 'undefined' && element.href === url) {
        document.head.removeChild(element);
      }
    });
  }

  /**
   * Append HTML to the body (default) or element with id containerId.
   *
   * @param content {string} Html content
   * @param containerId {string} ['body'] Container ID to append the HTML to. Defaults to body
   * element if omitted
   */
  addHtml(content, containerId = 'body') {
    let container = document.body;
    if (containerId !== 'body') {
      container = document.getElementById(containerId);
    }

    if (container === null) {
      console.warn(`Cannot add HTML to container #${containerId}!`);
      return;
    }

    let tmp       = document.createElement('div');
    tmp.innerHTML = content;

    while (tmp.firstChild) {
      container.appendChild(tmp.firstChild);
    }
  }

  /**
   * Append HTML to the body (default) or element with id containerId.
   *
   * @param id {string} Id for de-registering
   * @param content {string} Html content
   * @param containerId {string} ['body'] Container ID to append the HTML to. Defaults to body
   * element if omitted
   */
  registerHtml(id, content, containerId = 'body') {
    let container = document.body;
    if (containerId !== 'body') {
      container = document.getElementById(containerId);
    }

    if (container === null) {
      console.warn(`Cannot register #${id} into container #${containerId}!`);
      return;
    }

    const item = document.createElement('template');

    item.innerHTML = content;

    this._html.set(id, item.content);

    container.appendChild(item.content);
  }

  /**
   * Remove registered html from the body or container with id containerId.
   *
   * @see {registerHtml}
   * @param id {string}
   * @param containerId {string}
   */
  unregisterHtml(id, containerId = 'body') {
    let container = document.body;
    if (containerId !== 'body') {
      container = document.getElementById(containerId);
    }

    if (container === null) {
      console.warn(`Container #${containerId} does not exist!`);
      return;
    }

    if (!this._html.has(id)) {
      console.warn(`No Html with #${id} registered!`);
      return;
    }

    container.removeChild(this._html.get(id));
    this._html.delete(id);
  }

  /**
   * Tries to load the template content of 'id-template'.
   * Returns false if no template was found, HTMLElement otherwise.
   *
   * @param templateId {string} Template element id without '-template' suffix
   * @return {boolean|HTMLElement}
   */
  loadTemplateContent(templateId) {
    const id = `${templateId}-template`;

    if (this._templates.has(id)) {
      return this._templates.get(id).cloneNode(true).firstElementChild;
    }

    const template = document.getElementById(id);

    if (template === null) {
      console.warn(`Template '#${id}' not found!`);
      return false;
    }

    const {content} = template;
    this._templates.set(id, content);

    return content.cloneNode(true).firstElementChild;
  }

  /**
   * Clear the content of an element if it exists.
   *
   * @param element {string|HTMLElement} Element or ID
   * @return {void}
   */
  clearHtml(element) {
    if (typeof element === 'string') {
      // eslint-disable-next-line no-param-reassign
      element = document.getElementById(element);
    }

    if (element !== null && element instanceof HTMLElement) {
      while (element.firstChild !== null) {
        element.removeChild(element.firstChild);
      }
    } else {
      console.warn('Element could not be cleared.');
    }
  }

  /**
   * Toggle the class hidden on the given element.
   *
   * @param element {string|HTMLElement} Element or selector
   * @param element {string|HTMLElement} Element or ID
   * @return {void}
   */
  hideElement(element, hide) {
    if (typeof element === 'string') {
      // eslint-disable-next-line no-param-reassign
      element = document.querySelector(element);
    }

    if (element !== null && element instanceof HTMLElement) {
      if (hide) {
        element.classList.add('hidden');
      } else {
        element.classList.remove('hidden');
      }
    } else {
      console.warn('Element could not be shown / hidden!');
    }
  }

  /**
   * Initialize a noUiSlider.
   *
   * @param callbackName {string} tha data-callback attribute of the slider element
   * @param options {object} Options for noUiSlider
   */
  initSliderOptions(callbackName, options) {
    const slider = document.querySelector(`[data-callback="${callbackName}"]`);

    if (typeof noUiSlider === 'undefined') {
      console.warn('\'noUiSlider\' is not defined!');
      return;
    }

    noUiSlider.create(slider, options);

    var event = 'slide';
    if (slider.dataset.event) {
      event = slider.dataset.event;
    }

    slider.noUiSlider.on(event, (values, handle, unencoded) => {
      let callback = CosmoScout.callbacks.find(callbackName);
      if (callback !== undefined) {
        if (Array.isArray(unencoded)) {
          callback(unencoded[0], unencoded[1]);
        } else {
          callback(unencoded);
        }
      }
    });
  }

  /**
   * Initialize a non-linear noUiSlider.
   *
   * @param callbackName {string} tha data-callback attribute of the slider element
   * @param range {object} object defining the values at specific positions on the slider
   * @param start {number[]} Handle count and position
   */
  initSliderRange(callbackName, range, start) {
    this.initSliderOptions(callbackName, {
      start: start,
      connect: (start.length === 1 ? 'lower' : true),
      range: range,
      format: {
        to(value) {
          return CosmoScout.utils.beautifyNumber(value);
        },
        from(value) {
          return Number(parseFloat(value));
        },
      },
    });
  }

  /**
   * Initialize a linear noUiSlider.
   *
   * @param callbackName {string} tha data-callback attribute of the slider element
   * @param min {number} Min value
   * @param max {number} Max value
   * @param step {number} Step size
   * @param start {number[]} Handle count and position
   */
  initSlider(callbackName, min, max, step, start) {
    this.initSliderOptions(callbackName, {
      start: start,
      connect: (start.length === 1 ? 'lower' : true),
      step: step,
      range: {min: min, max: max},
      format: {
        to(value) {
          return CosmoScout.utils.beautifyNumber(value);
        },
        from(value) {
          return Number(parseFloat(value));
        },
      }
    });
  }

  /**
   * Sets a noUiSlider value.
   *
   * @param callbackName {string} tha data-callback attribute of the slider element
   * @param value {number} Value
   */
  setSliderValue(callbackName, emitCallbacks, ...value) {
    const slider = document.querySelector(`[data-callback="${callbackName}"]`);

    if (slider !== null && typeof slider.noUiSlider !== 'undefined') {
      if (!slider.matches(":active")) {
        if (value.length === 1) {
          slider.noUiSlider.set(value[0], emitCallbacks);
        } else {
          slider.noUiSlider.set(value, emitCallbacks);
        }
      }
    } else {
      console.warn(`Slider '${callbackName} 'not found or 'noUiSlider' not active.`);
    }
  }

  /**
   * Clears the content of a selecticker dropdown.
   *
   * @param callbackName {string} tha data-callback attribute of the dropdown element
   */
  clearDropdown(callbackName) {
    const dropdown = document.querySelector(`[data-callback="${callbackName}"]`);
    CosmoScout.gui.clearHtml(dropdown);

    $(dropdown).selectpicker('refresh');
  }

  /**
   * Adds an option to a dropdown.
   *
   * @param callbackName {string} tha data-callback attribute of the dropdown element
   * @param value {string|number} Option value
   * @param text {string} Option text
   * @param selected {boolean|string} Selected flag
   */
  addDropdownValue(callbackName, value, text, selected = false) {
    const dropdown = document.querySelector(`[data-callback="${callbackName}"]`);
    const option   = document.createElement('option');

    option.value       = value;
    option.selected    = selected ? true : false;
    option.textContent = text;

    if (dropdown !== null) {
      dropdown.appendChild(option);

      $(dropdown).selectpicker('refresh');
    } else {
      console.warn(`Dropdown '${callbackName} 'not found`);
    }
  }

  /**
   * Sets the current value of a selectpicker.
   *
   * @param callbackName {string} tha data-callback attribute of the dropdown element
   * @param value {string|number}
   */
  setDropdownValue(callbackName, value, emitCallbacks) {
    const dropdown = document.querySelector(`[data-callback="${callbackName}"]`);
    $(dropdown).selectpicker('val', value);

    if (emitCallbacks) {
      this._emitChangeEvent(dropdown);
    }
  }

  /**
   * Sets a radio button to checked.
   *
   * @see {setCheckboxValue}
   * @param callbackName {string} tha data-callback attribute of the radio button element
   */
  setRadioChecked(callbackName, emitCallbacks) {
    this.setCheckboxValue(callbackName, true, emitCallbacks);
  }

  /**
   * Sets a checkboxs checked state to true/false.
   *
   * @param callbackName {string} tha data-callback attribute of the radio button element
   * @param value {boolean} True = checked / False = unchecked
   */
  setCheckboxValue(callbackName, value, emitCallbacks) {
    const element = document.querySelector(`[data-callback="${callbackName}"]`);

    if (element !== null) {
      element.checked = value;

      if (emitCallbacks) {
        this._emitChangeEvent(element);
      }
    }
  }

  /**
   * Sets the value of a text input.
   * Only selects .text-input s which descend .item-ID
   *
   * @param callbackName {string} tha data-callback attribute of the text input element
   * @param value {string}
   */
  setTextboxValue(id, value) {
    const element = document.querySelector(`.item-${id} .text-input`);

    if (element !== null) {
      element.value = value;
    }
  }

  /**
   * Triggers an artificial change event on a given HTML element.
   *
   * @param element {HTMLElement} The element to fire the event on
   * @private
   */
  _emitChangeEvent(element) {
    let evt = document.createEvent("HTMLEvents");
    evt.initEvent("change", false, true);
    element.dispatchEvent(evt);
  }
}
