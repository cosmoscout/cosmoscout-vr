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
   * Divs of the Bookmark Editor.
   * @type {HTMLElement}
   */
  _editor;
  _saveButton;
  _nothingGivenError;
  _nameDiv;
  _descriptionDiv;
  _colorDiv;
  _startDateDiv;
  _endDateDiv;
  _centerDiv;
  _frameDiv;
  _locationXDiv;
  _locationYDiv;
  _locationZDiv;
  _rotationXDiv;
  _rotationYDiv;
  _rotationZDiv;
  _rotationWDiv;

  /**
   * @inheritDoc
   */
  init() {
    this._editor            = document.querySelector("#bookmark-editor");
    this._nothingGivenError = document.querySelector("#bookmark-editor-nothing-given-error");
    this._saveButton        = document.querySelector("#bookmark-editor-save-button");
    this._nameDiv           = document.querySelector("#bookmark-editor-name");
    this._descriptionDiv    = document.querySelector("#bookmark-editor-description");
    this._colorDiv          = document.querySelector("#bookmark-editor-color");
    this._startDateDiv      = document.querySelector("#bookmark-editor-start-date");
    this._endDateDiv        = document.querySelector("#bookmark-editor-end-date");
    this._centerDiv         = document.querySelector("#bookmark-editor-center");
    this._frameDiv          = document.querySelector("#bookmark-editor-frame");
    this._locationXDiv      = document.querySelector("#bookmark-editor-location-x");
    this._locationYDiv      = document.querySelector("#bookmark-editor-location-y");
    this._locationZDiv      = document.querySelector("#bookmark-editor-location-z");
    this._rotationXDiv      = document.querySelector("#bookmark-editor-rotation-x");
    this._rotationYDiv      = document.querySelector("#bookmark-editor-rotation-y");
    this._rotationZDiv      = document.querySelector("#bookmark-editor-rotation-z");
    this._rotationWDiv      = document.querySelector("#bookmark-editor-rotation-w");

    // Connect buttons setting fields to current values --------------------------------------------

    document.querySelector("#bookmark-editor-start-date + div > button").onmouseup = () => {
      this._startDateDiv.value = CosmoScout.state.simulationTime.toISOString();
    };

    document.querySelector("#bookmark-editor-end-date + div > button").onmouseup = () => {
      this._endDateDiv.value = CosmoScout.state.simulationTime.toISOString();
    };

    document.querySelector("#bookmark-editor-frame + div > button").onmouseup = () => {
      this._centerDiv.value = CosmoScout.state.activePlanetCenter;
      this._frameDiv.value  = CosmoScout.state.activePlanetFrame;
    };

    document.querySelector("#bookmark-editor-location-z + div > button").onmouseup = () => {
      this._locationXDiv.value = CosmoScout.state.observerPosition[0];
      this._locationYDiv.value = CosmoScout.state.observerPosition[1];
      this._locationZDiv.value = CosmoScout.state.observerPosition[2];
    };

    document.querySelector("#bookmark-editor-rotation-w + div > button").onmouseup = () => {
      this._rotationXDiv.value = CosmoScout.state.observerRotation[0];
      this._rotationYDiv.value = CosmoScout.state.observerRotation[1];
      this._rotationZDiv.value = CosmoScout.state.observerRotation[2];
      this._rotationWDiv.value = CosmoScout.state.observerRotation[3];
    };

    // Save bookmark on save button click ----------------------------------------------------------

    this._saveButton.onmouseup = () => {
      // First validate all fields.
      let nameGiven      = this._nameDiv.value != "";
      let startDateGiven = this._startDateDiv.value != "";
      let endDateGiven   = this._endDateDiv.value != "";
      let centerGiven    = this._centerDiv.value != "";
      let frameGiven     = this._frameDiv.value != "";

      let anyLocationGiven = this._locationXDiv.value != "" || this._locationYDiv.value != "" ||
                             this._locationZDiv.value != "";

      let anyRotationGiven = this._rotationXDiv.value != "" || this._rotationYDiv.value != "" ||
                             this._rotationZDiv.value != "" || this._rotationWDiv.value != "";

      let centerRequired   = anyRotationGiven || anyLocationGiven || frameGiven;
      let frameRequired    = anyRotationGiven || anyLocationGiven || centerGiven;
      let locationRequired = anyRotationGiven || anyLocationGiven;
      let rotationRequired = anyRotationGiven;

      let highlight = (div, doHighlight) => {
        if (doHighlight) {
          div.classList.add("is-invalid");
        } else {
          div.classList.remove("is-invalid");
        }
      };

      highlight(this._nameDiv, !nameGiven);
      highlight(this._startDateDiv, endDateGiven && !startDateGiven);
      highlight(this._centerDiv, !centerGiven && centerRequired);
      highlight(this._frameDiv, !frameGiven && frameRequired);
      highlight(this._locationXDiv, locationRequired && this._locationXDiv.value == "");
      highlight(this._locationYDiv, locationRequired && this._locationYDiv.value == "");
      highlight(this._locationZDiv, locationRequired && this._locationZDiv.value == "");
      highlight(this._rotationXDiv, rotationRequired && this._rotationXDiv.value == "");
      highlight(this._rotationYDiv, rotationRequired && this._rotationYDiv.value == "");
      highlight(this._rotationZDiv, rotationRequired && this._rotationZDiv.value == "");
      highlight(this._rotationWDiv, rotationRequired && this._rotationWDiv.value == "");

      if (nameGiven && !(startDateGiven || endDateGiven || centerGiven || frameGiven ||
                           anyLocationGiven || anyRotationGiven)) {
        this._nothingGivenError.style.display = "block";
      } else {
        this._nothingGivenError.style.display = "none";
      }
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
