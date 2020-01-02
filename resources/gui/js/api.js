/* global $,Format,noUiSlider */
/* eslint max-classes-per-file: 0 */
'use strict';

/**
 * Simplistic api interface containing a name field and init method
 */
// eslint-disable-next-line no-unused-vars
class IApi {
  /**
   * Api Name
   *
   * @type {string}
   */
  name;

  /**
   * Called in CosmoScout.init
   */
  // eslint-disable-next-line class-methods-use-this
  init() {
  }

  /**
   * Replace common template markers with content
   *
   * @param html {string}
   * @param id {string}
   * @param icon {string}
   * @param content {string}
   * @return {string}
   * @protected
   */
  replaceMarkers(html, id, icon, content) {
    return html
      .replace(this.regex('ID'), id)
      .replace(this.regex('CONTENT'), content)
      .replace(this.regex('ICON'), icon)
      .trim();
  }

  /**
   * Creates a search global Regex Object of %matcher%
   *
   * @param matcher {string}
   * @return {RegExp}
   * @protected
   */
  // eslint-disable-next-line class-methods-use-this
  regex(matcher) {
    return new RegExp(`%${matcher}%`, 'g');
  }
}

/**
 * Api Container holding all registered apis.
 */
// eslint-disable-next-line no-unused-vars
class CosmoScout {
  /**
   * @type {Map<string, Object>}
   * @private
   */
  static _apis = new Map();

  /**
   * Cache loaded templates
   *
   * @type {Map<string, DocumentFragment>}
   * @private
   */
  static _templates = new Map();

  /**
   * Registered html parts
   *
   * @see {registerHtml}
   * @type {Map<string, DocumentFragment>}
   * @private
   */
  static _html = new Map();

  /**
   * Init a list of apis
   *
   * @param apis {IApi}
   */
  static init(...apis) {
    [...apis].forEach((Api) => {
      try {
        let instance;

        if (typeof Api === 'string' && String(Api).slice(-3) === 'Api') {
          // eslint-disable-next-line no-eval
          instance = eval(`new ${Api}()`);
        } else {
          instance = new Api();
        }

        this.register(instance.name, instance);
        instance.init();
      } catch (e) {
        console.error(`Could not initialize ${Api}`);
      }
    });
  }

  /**
   * Initialize third party drop downs,
   * add input event listener,
   * initialize tooltips
   */
  static initInputs() {
    this.initDropDowns();
    this.initChecklabelInputs();
    this.initRadiolabelInputs();
    this.initTooltips();
    this.initDataCalls();
  }

  /**
   * @see {initInputs}
   * TODO Remove jQuery
   */
  static initDropDowns() {
    const dropdowns = $('.simple-value-dropdown');
    dropdowns.selectpicker();

    const eventListener = (event) => {
      if (event.target !== null && event.target.id !== '') {
        CosmoScout.callNative(event.target.id, event.target.value);
      }
    };

    document.querySelectorAll('.simple-value-dropdown').forEach((dropdown) => {
      dropdown.addEventListener('change', eventListener);
    });
  }

  /**
   * @see {initInputs}
   */
  static initChecklabelInputs() {
    document.querySelectorAll('.checklabel input').forEach((input) => {
      if (typeof input.dataset.initialized !== 'undefined') {
        return;
      }

      input.addEventListener('change', (event) => {
        if (event.target !== null) {
          CosmoScout.callNative(event.target.id, event.target.checked);
        }
      });

      input.dataset.initialized = 'true';
    });
  }

  /**
   * @see {initInputs}
   */
  static initRadiolabelInputs() {
    document.querySelectorAll('.radiolabel input').forEach((input) => {
      if (typeof input.dataset.initialized !== 'undefined') {
        return;
      }

      input.addEventListener('change', (event) => {
        if (event.target !== null) {
          CosmoScout.callNative(event.target.id);
        }
      });

      input.dataset.initialized = 'true';
    });
  }

  /**
   * @see {initInputs}
   * @see {callNative}
   * Adds an onclick listener to every element containing [data-call="'methodname'"]
   * The method name gets passed to CosmoScout.callNative.
   * Arguments can be passed by separating the content with ','
   * E.g.: fly_to,Africa -> CosmoScout.callNative('fly_to', 'Africa')
   *       method,arg1,...,argN -> CosmoScout.callNative('method', arg1, ..., argN)
   */
  static initDataCalls() {
    document.querySelectorAll('[data-call]').forEach((input) => {
      if (typeof input.dataset.initialized !== 'undefined') {
        return;
      }

      input.addEventListener('click', () => {
        if (typeof input.dataset.call !== 'undefined') {
          const args = input.dataset.call;

          eval(`CosmoScout.callNative(${args})`)
        }
      });

      input.dataset.initialized = 'true';
    });
  }

  /**
   * @see {initInputs}
   */
  static initTooltips() {
    const config = { delay: 500, placement: 'auto', html: false };

    /* Boostrap Tooltips require jQuery for now */
    $('[data-toggle="tooltip"]').tooltip(config);
    config.placement = 'bottom';
    $('[data-toggle="tooltip-bottom"]').tooltip(config);
  }

  /**
   * Appends a script element to the body
   *
   * @param url {string} Absolute or local file path
   * @param init {Function} Method gets run on script load
   */
  static registerJavaScript(url, init) {
    const script = document.createElement('script');
    script.setAttribute('type', 'text/javascript');
    script.setAttribute('src', url);

    if (typeof init !== 'undefined') {
      script.addEventListener('load', init);
      script.addEventListener('readystatechange', init);
    }

    document.body.appendChild(script);
  }

  /**
   * Removes a script element by url
   *
   * @param url {string}
   */
  static unregisterJavaScript(url) {
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
  static registerCss(url) {
    const link = document.createElement('link');
    link.setAttribute('type', 'text/css');
    link.setAttribute('rel', 'stylesheet');
    link.setAttribute('href', url);

    document.head.appendChild(link);
  }

  /**
   * Removes a stylesheet by url
   *
   * @param url {string}
   */
  static unregisterCss(url) {
    document.querySelectorAll('link').forEach((element) => {
      if (typeof element.href !== 'undefined'
        && (element.href === url || element.href === this._localizeUrl(url))) {
        document.head.removeChild(element);
      }
    });
  }

  /**
   * Append HTML to body per default or element with id containerId
   *
   * @param id {string}
   * @param content {string}
   * @param containerId {string}
   */
  static registerHtml(id, content, containerId = 'body') {
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
   * Remove registered html from the body of container with id containerId
   *
   * @param id {string}
   * @param containerId {string}
   */
  static unregisterHtml(id, containerId = 'body') {
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
  static loadTemplateContent(templateId) {
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
  static clearHtml(element) {
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
   * @param id {string}
   * @param min {number}
   * @param max {number}
   * @param step {number}
   * @param start {number[]}
   */
  static initSlider(id, min, max, step, start) {
    const slider = document.getElementById(id);

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
          return Format.beautifyNumber(value);
        },
        from(value) {
          return Number(parseFloat(value));
        },
      },
    });

    slider.noUiSlider.on('slide', (values, handle, unencoded) => {
      if (Array.isArray(unencoded)) {
        CosmoScout.callNative(id, unencoded[handle], handle);
      } else {
        CosmoScout.callNative(id, unencoded, 0);
      }
    });
  }

  /**
   * Set a noUiSlider value
   *
   * @param id {string} Slider ID
   * @param value {number} Value
   */
  static setSliderValue(id, ...value) {
    const slider = document.getElementById(id);

    if (slider !== null && typeof slider.noUiSlider !== 'undefined') {
      if (value.length === 1) {
        slider.noUiSlider.set(value[0]);
      } else {
        slider.noUiSlider.set(value);
      }
    } else {
      console.warn(`Slider '${id} 'not found or 'noUiSlider' not active.`);
    }
  }

  /**
   * @param id {string}
   */
  static clearDropdown(id) {
    CosmoScout.clearHtml(id);

    $(`#${id}`).selectpicker('render');
  }

  /**
   * Adds an option to a dropdown
   * TODO remove jQuery
   *
   * @param id {string} DropDown ID
   * @param value {string|number} Option value
   * @param text {string} Option text
   * @param selected {boolean} Selected flag
   */
  static addDropdownValue(id, value, text, selected = false) {
    const dropdown = document.getElementById(id);
    const option = document.createElement('option');

    option.value = value;
    option.selected = selected === true;
    option.text = text;

    if (dropdown !== null) {
      dropdown.appendChild(option);

      $(`#${id}`).selectpicker('refresh');
    } else {
      console.warn(`Dropdown '${id} 'not found`);
    }
  }

  /**
   * @param id {string}
   * @param value {string|number}
   */
  static setDropdownValue(id, value) {
    $(`#${id}`).selectpicker('val', value);
  }

  /**
   * @param id {string}
   */
  static setRadioChecked(id) {
    this.setCheckboxValue(id, true);
  }

  /**
   * @param id {string}
   * @param value {boolean}
   */
  static setCheckboxValue(id, value) {
    const element = document.getElementById(id);

    if (element !== null) {
      element.checked = value === true;
    }
  }

  /**
   * @param id {string}
   * @param value {string}
   */
  static setTextboxValue(id, value) {
    const element = document.querySelector(`.item-${id} .text-input`);

    if (element !== null) {
      element.value = value;
    }
  }

  /**
   * window.call_native wrapper
   *
   * @param fn {string}
   * @param args {any}
   * @return {*}
   */
  static callNative(fn, ...args) {
    return window.call_native(fn, ...args);
  }

  /**
   * Register an api object
   *
   * @param name {string}
   * @param api {Object}
   */
  static register(name, api) {
    this[name] = api;
    this._apis.set(name, api);
  }

  /**
   * Remove a registered api by name
   *
   * @param name {string}
   */
  static remove(name) {
    delete this[name];
    this._apis.delete(name);
  }

  /**
   * Get a registered api object
   *
   * @param name {string}
   * @return {Object}
   */
  static getApi(name) {
    return this._apis.get(name);
  }

  /**
   * Localizes a filename
   *
   * @param url {string}
   * @return {string}
   * @private
   */
  static _localizeUrl(url) {
    return `file://../share/resources/gui/${url}`;
  }
}
