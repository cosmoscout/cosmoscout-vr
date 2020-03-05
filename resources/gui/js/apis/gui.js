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
  }

  /**
   * Initializes all .simple-value-dropdown with bootstrap's selectpicker.
   *
   * @see {initInputs}
   */
  initDropDowns() {
    const dropdowns = $('.simple-value-dropdown');
    dropdowns.selectpicker();

    const eventListener = (event) => {
      if (event.target !== null) {
        let callback = CosmoScout.callbacks.find(event.target.dataset.callback)

        if (callback !== undefined) {
          callback(event.target.value);
        }
      }
    };

    document.querySelectorAll('.simple-value-dropdown').forEach((dropdown) => {
      if (typeof dropdown.dataset.initialized !== 'undefined') {
        return;
      }

      dropdown.addEventListener('change', eventListener);
    });
  }

  /**
   * Adds a 'change' event listener which calls callNative with id and checkstate
   * Will only add a listener once
   *
   * @see {callNative}
   * @see {initInputs}
   */
  initChecklabelInputs() {
    document.querySelectorAll('.checklabel input').forEach((input) => {
      if (typeof input.dataset.initialized !== 'undefined') {
        return;
      }

      input.addEventListener('change', (event) => {
        if (event.target !== null) {
          let callback = CosmoScout.callbacks.find(event.target.dataset.callback)

          if (callback !== undefined) {
            callback(event.target.checked);
          }
        }
      });

      input.dataset.initialized = 'true';
    });
  }

  /**
   * Adds a change event listener which calls callNative with the target id
   *
   * @see {callNative}
   * @see {initInputs}
   */
  initRadiolabelInputs() {
    document.querySelectorAll('.radiolabel input').forEach((input) => {
      if (typeof input.dataset.initialized !== 'undefined') {
        return;
      }

      input.addEventListener('change', (event) => {
        if (event.target !== null) {
          let callback = CosmoScout.callbacks.find(event.target.dataset.callback)

          if (callback !== undefined) {
            callback(event.target.checked);
          }
        }
      });

      input.dataset.initialized = 'true';
    });
  }

  /**
   * Initializes [data-toggle="tooltip"] elements.
   *
   * @see {initInputs}
   */
  initTooltips() {
    const config = { delay: 500, placement: 'auto', html: false };

    /* Bootstrap Tooltips require jQuery for now */
    $('[data-toggle="tooltip"]').tooltip(config);
    config.placement = 'bottom';
    $('[data-toggle="tooltip-bottom"]').tooltip(config);
  }

  /**
   * Appends a script element to the body
   *
   * @param url {string} Absolute or local file path
   * @param init {string|Function} Method gets run on script load
   */
  registerJavaScript(url, init) {
    const script = document.createElement('script');

    if (typeof init !== 'undefined') {
      if (typeof init === 'string') {
        'use strict';
        init = eval(init);
      }

      script.addEventListener('readystatechange', init);
    }

    script.setAttribute('src', url);

    document.body.appendChild(script);
  }

  /**
   * Removes a script element by url
   *
   * @param url {string}
   */
  unregisterJavaScript(url) {
    document.querySelectorAll('script').forEach((element) => {
      if (typeof element.src !== 'undefined'
        && (element.src === url || element.src === this._localizeUrl(url))) {
        document.body.removeChild(element);
      }
    });
  }

  /**
   * Appends a link stylesheet to the head
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
   * Removes a stylesheet by url
   *
   * @param url {string}
   */
  unregisterCss(url) {
    document.querySelectorAll('link').forEach((element) => {
      if (typeof element.href !== 'undefined'
        && (element.href === url || element.href === this._localizeUrl(url))) {
        document.head.removeChild(element);
      }
    });
  }

  /**
   * Append HTML to the body (default) or element with id containerId
   *
   * @param id {string} Id for de-registering
   * @param content {string} Html content
   * @param containerId {string} ['body'] Container ID to append the HTML to. Defaults to body element if omitted
   */
  registerHtml(id, content, containerId = 'body') {
    let container = document.body;
    if (containerId !== 'body') {
      container = document.getElementById(containerId);
    }

    if (container === null) {
      console.error(`Cannot register #${id} into container #${containerId}.`);
      return;
    }

    const item = document.createElement('template');

    item.innerHTML = content;

    this._html.set(id, item.content);

    container.appendChild(item.content);
  }

  /**
   * Remove registered html from the body or container with id containerId
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
      console.error(`Container #${containerId} does not exist.`);
      return;
    }

    if (!this._html.has(id)) {
      console.error(`No Html with #${id} registered.`);
      return;
    }

    container.removeChild(this._html.get(id));
    this._html.delete(id);
  }

  /**
   * Tries to load the template content of 'id-template'
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
      console.error(`Template '#${id}' not found.`);
      return false;
    }

    const { content } = template;
    this._templates.set(id, content);

    return content.cloneNode(true).firstElementChild;
  }

  /**
   * Clear the content of an element if it exists
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
   * Initialize a noUiSlider
   *
   * @param id {string} Slider html id without '#'
   * @param min {number} Min value
   * @param max {number} Max value
   * @param step {number} Step size
   * @param start {number[]} Handle count and position
   */
  initSlider(callbackName, min, max, step, start) {
    const slider = document.querySelector(`[data-callback="${callbackName}"]`);

    if (typeof noUiSlider === 'undefined') {
      console.error('\'noUiSlider\' is not defined.');
      return;
    }

    noUiSlider.create(slider, {
      start,
      connect: (start.length === 1 ? 'lower' : true),
      step,
      range: { min, max },
      format: {
        to(value) {
          return CosmoScout.utils.beautifyNumber(value);
        },
        from(value) {
          return Number(parseFloat(value));
        },
      },
    });

    slider.noUiSlider.on('slide', (values, handle, unencoded) => {
      let callback = CosmoScout.callbacks.find(callbackName);
      if (callback !== undefined) {
        if (Array.isArray(unencoded)) {
          callback(unencoded[handle], handle);
        } else {
          callback(unencoded, 0);
        }
      }
    });
  }

  /**
   * Sets a noUiSlider value
   *
   * @param id {string} Slider ID
   * @param value {number} Value
   */
  setSliderValue(callbackName, ...value) {
    const slider = document.querySelector(`[data-callback="${callbackName}"]`);

    if (slider !== null && typeof slider.noUiSlider !== 'undefined') {
      if (value.length === 1) {
        slider.noUiSlider.set(value[0]);
      } else {
        slider.noUiSlider.set(value);
      }
    } else {
      console.warn(`Slider '${callbackName} 'not found or 'noUiSlider' not active.`);
    }
  }

  /**
   * Clears the content of a selecticker dropdown
   *
   * @param id {string}
   */
  clearDropdown(callbackName) {
    const dropdown = document.querySelector(`[data-callback="${callbackName}"]`);
    CosmoScout.gui.clearHtml(dropdown);

    $(dropdown).selectpicker('render');
  }

  /**
   * Adds an option to a dropdown
   * TODO remove jQuery
   *
   * @param id {string} DropDown ID
   * @param value {string|number} Option value
   * @param text {string} Option text
   * @param selected {boolean|string} Selected flag
   */
  addDropdownValue(callbackName, value, text, selected = false) {
    const dropdown = document.querySelector(`[data-callback="${callbackName}"]`);
    const option = document.createElement('option');

    option.value = value;
    option.selected = selected ? true : false;
    option.textContent = text;

    if (dropdown !== null) {
      dropdown.appendChild(option);

      $(dropdown).selectpicker('refresh');
    } else {
      console.warn(`Dropdown '${callbackName} 'not found`);
    }
  }

  /**
   * Sets the current value of a selectpicker
   *
   * @param id {string}
   * @param value {string|number}
   */
  setDropdownValue(callbackName, value) {
    const dropdown = document.querySelector(`[data-callback="${callbackName}"]`);
    $(dropdown).selectpicker('val', value);
  }

  /**
   * Sets a radio button to checked
   *
   * @see {setCheckboxValue}
   * @param id {string} Radiobutton id
   */
  setRadioChecked(callbackName) {
    this.setCheckboxValue(callbackName, true);
  }

  /**
   * Sets a checkboxs checked state to true/false
   *
   * @param id {string} Checkbox id
   * @param value {boolean} True = checked / False = unchecked
   */
  setCheckboxValue(callbackName, value) {
    const element = document.querySelector(`[data-callback="${callbackName}"]`);

    if (element !== null) {
      element.checked = value === true;
    }
  }

  /**
   * Sets the value of a text input
   * Only selects .text-input s which descend .item-ID
   *
   * @param id {string}
   * @param value {string}
   */
  setTextboxValue(id, value) {
    const element = document.querySelector(`.item-${id} .text-input`);

    if (element !== null) {
      element.value = value;
    }
  }

  /**
   * Localizes a filename
   *
   * @param url {string}
   * @return {string}
   * @private
   */
  _localizeUrl(url) {
    return `file://../share/resources/gui/${url}`;
  }
}
