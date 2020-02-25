/* global $,Format,noUiSlider */
/* eslint-disable max-classes-per-file */

/**
 * Api Container holding all registered apis.
 */
// eslint-disable-next-line no-unused-vars
class CosmoScout {
  /**
   * Registered apis
   *
   * @see {IApi.name}
   * @type {Map<string, Object>}
   * @private
   */
  static _apis = new Map();

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

        if (instance.name === '') {
          console.error(`${instance.constructor.name} is missing the 'name' property.`);
          return;
        }

        this.register(instance.name, instance);
        instance.init();
      } catch (e) {
        console.error(`Could not initialize ${Api}: ${e.message}`);
      }
    });
  }

  /**
   * This is called once a frame.
   *
   * @param apis {IApi}
   */
  static update() {
    this._apis.forEach(api => api.update());
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
   * @param name {string} Api name from IApi
   * @param api {Object} Instantiated IApi object
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
}
