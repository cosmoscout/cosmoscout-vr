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

  /**
   * Replace common template markers with content
   *
   * @param html {string} HTML with %MARKER% markers
   * @param id {string} Id marker replacement
   * @param icon {string} Icon marker replacement
   * @param content {string} Content marker replacement
   * @return {string} replaced html
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
   * Creates a search global Regex Object of %MATCHER%
   *
   * @param matcher {string}
   * @return {RegExp}
   * @protected
   */
  // eslint-disable-next-line class-methods-use-this
  regex(matcher) {
    return new RegExp(`%${String(matcher).toUpperCase()}%`, 'g');
  }
}
