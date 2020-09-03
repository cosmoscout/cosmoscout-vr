/* global $,Format,noUiSlider */
/* eslint-disable max-classes-per-file */

/**
 * Api Container holding all registered apis.
 */
// eslint-disable-next-line no-unused-vars
class CosmoScoutAPI {

  /**
   * Stores all callbacks registered via C++. It has one default "callbacks.find()" method, which
   * can be used to call callbacks which are actually registered as sub objects.
   * callbacks.find("notifications.print") will return the function "print" registered on
   * the object "notifications".
   */
  callbacks = {
    find: (name) => {
      try {
        let callback = name.split('.').reduce((a, b) => a[b], CosmoScout.callbacks);
        if (callback !== undefined) {
          return callback;
        }
      } catch (e) {}

      console.warn(`Failed to find callback ${name} on CosmoScout.callbacks!`);

      return null;
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
  state = {};

  /**
   * Registered apis.
   *
   * @see {IApi.name}
   * @type {Map<string, Object>}
   * @private
   */
  _apis = new Map();

  /**
   * Init a list of apis.
   *
   * @param apis {IApi}
   */
  init(...apis) {
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
      } catch (e) { console.error(`Could not initialize ${Api}: ${e.message}`); }
    });
  }

  /**
   * This is called once a frame.
   *
   * @param apis {IApi}
   */
  update() {
    this._apis.forEach(api => api.update());
  }

  /**
   * Register an api object.
   *
   * @param name {string} Api name from IApi
   * @param api {Object} Instantiated IApi object
   */
  registerApi(name, api) {
    this._apis.set(name, api);
    this[name] = api;
  }

  /**
   * Remove a registered api by name
   *
   * @param name {string}
   */
  removeApi(name) {
    this._apis.delete(name);
    delete this[name];
  }
}
