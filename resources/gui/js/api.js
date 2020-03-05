/* global $,Format,noUiSlider */
/* eslint-disable max-classes-per-file */

/**
 * When you create a plugin for CosmoScout VR, you can derive from this class. Your JavaScript
 * code can be initialized in init() and updated regularly within update().
 */
class IApi {
  /**
   * Api Name. Once registered via CosmoScoutAPI.init(), methods of the derived class can be called
   *  with CosmoScout.<api name>.<method name>()
   *
   * @type {string}
   */
  name;

  /**
   * Called when the API is registered via CosmoScout.init()
   */
  init() {
  }

  /**
   * Automatically called once a frame. You should override this if you want to do something at
   * regular intervals.
   */
  update() {
  }
}
