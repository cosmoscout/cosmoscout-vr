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

    document.querySelector("#bookmark-editor-start-date + div > button").onmouseup = () => {
      document.querySelector("#bookmark-editor-start-date").value = CosmoScout.state.simulationTime;
    };

    document.querySelector("#bookmark-editor-end-date + div > button").onmouseup = () => {
      document.querySelector("#bookmark-editor-end-date").value = CosmoScout.state.simulationTime;
    };

    document.querySelector("#bookmark-editor-frame + div > button").onmouseup = () => {
      document.querySelector("#bookmark-editor-center").value = CosmoScout.state.activePlanetCenter;
      document.querySelector("#bookmark-editor-frame").value  = CosmoScout.state.activePlanetFrame;
    };

    document.querySelector("#bookmark-editor-location-z + div > button").onmouseup = () => {
      document.querySelector("#bookmark-editor-location-x").value =
          CosmoScout.state.observerPosition[0];
      document.querySelector("#bookmark-editor-location-y").value =
          CosmoScout.state.observerPosition[1];
      document.querySelector("#bookmark-editor-location-z").value =
          CosmoScout.state.observerPosition[2];
    };

    document.querySelector("#bookmark-editor-rotation-w + div > button").onmouseup = () => {
      document.querySelector("#bookmark-editor-rotation-x").value =
          CosmoScout.state.observerRotation[0];
      document.querySelector("#bookmark-editor-rotation-y").value =
          CosmoScout.state.observerRotation[1];
      document.querySelector("#bookmark-editor-rotation-z").value =
          CosmoScout.state.observerRotation[2];
      document.querySelector("#bookmark-editor-rotation-w").value =
          CosmoScout.state.observerRotation[3];
    };
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
