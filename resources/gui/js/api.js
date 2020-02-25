/* global $,Format,noUiSlider */
/* eslint-disable max-classes-per-file */

/**
 * Simplistic api interface containing a name field and init method
 */
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
  init() {
  }

  /**
   * Automatically called once a frame
   */
  update() {
  }
}
