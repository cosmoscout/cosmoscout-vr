/* global IApi, CosmoScout */

/**
 * The loading screen
 */
class BookmarksApi extends IApi {
  /**
   * @inheritDoc
   */
  name = 'bookmarks';

  /**
   * @type {HTMLElement}
   */
  _editor;

  /**
   * @inheritDoc
   */
  init() {
    this._editor = document.querySelector("#bookmark-editor");
  }

  /**
   * Sets the visibility of the Bookmark Editor to the given value (true or false).
   *
   * @param visible {boolean}
   */
  setVisible(visible) {
    if (visible) {
      this._editor.classList.add("visible");
    } else {
      this._editor.classList.remove("visible");
    }
  }

  /**
   * Toggles the visibility of the Bookmark Editor.
   */
  toggle() {
    this.setVisible(!this._editor.classList.contains("visible"));
  }
}
