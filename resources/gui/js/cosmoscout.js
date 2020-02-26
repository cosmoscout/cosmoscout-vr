/* global $,Format,noUiSlider */
/* eslint-disable max-classes-per-file */

/**
 * Api Container holding all registered apis.
 */
// eslint-disable-next-line no-unused-vars
class CosmoScout {
  
  /**
   * Stores all callbacks registered via C++
   */
  static callbacks = {
    find: (name) => {
      try {
        let callback = name.split('.').reduce((a, b) => a[b], CosmoScout.callbacks);
        if (callback !== undefined) {
          return callback;
        }
      } catch(e) {}

      console.warn(`Failed to find callback ${name} on CosmoScout.callbacks!`);
    }
  };

  /**
   * Use this to access read-only state variables which are set from C++.
   * activePlanetCenter
   * activePlanetFrame
   * observerSpeed
   * pointerPosition
   * observerPosition
   */
  static state = {};

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

        this.registerApi(instance.name, instance);
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
   * Register an api object
   *
   * @param name {string} Api name from IApi
   * @param api {Object} Instantiated IApi object
   */
  static registerApi(name, api) {
    this[name] = api;
    this._apis.set(name, api);
  }

  /**
   * Remove a registered api by name
   *
   * @param name {string}
   */
  static removeApi(name) {
    delete this[name];
    this._apis.delete(name);
  }
}
