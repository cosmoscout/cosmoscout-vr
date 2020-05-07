/* global IApi, CosmoScout */

/**
 * The loading screen
 */
class BookmarkEditorApi extends IApi {
  /**
   * @inheritDoc
   */
  name = 'bookmarkEditor';

  /**
   * Divs of the Bookmark Editor.
   * @type {HTMLElement}
   */
  _title;
  _editor;
  _saveButton;
  _deleteButton;
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

  _editBookmarkID = null;

  /**
   * @inheritDoc
   */
  init() {
    this._title             = document.getElementById("bookmark-editor-title");
    this._editor            = document.getElementById("bookmark-editor");
    this._nothingGivenError = document.getElementById("bookmark-editor-nothing-given-error");
    this._saveButton        = document.getElementById("bookmark-editor-save-button");
    this._deleteButton      = document.getElementById("bookmark-editor-delete-button");
    this._nameDiv           = document.getElementById("bookmark-editor-name");
    this._descriptionDiv    = document.getElementById("bookmark-editor-description");
    this._colorDiv          = document.getElementById("bookmark-editor-color");
    this._startDateDiv      = document.getElementById("bookmark-editor-start-date");
    this._endDateDiv        = document.getElementById("bookmark-editor-end-date");
    this._centerDiv         = document.getElementById("bookmark-editor-center");
    this._frameDiv          = document.getElementById("bookmark-editor-frame");
    this._locationXDiv      = document.getElementById("bookmark-editor-location-x");
    this._locationYDiv      = document.getElementById("bookmark-editor-location-y");
    this._locationZDiv      = document.getElementById("bookmark-editor-location-z");
    this._rotationXDiv      = document.getElementById("bookmark-editor-rotation-x");
    this._rotationYDiv      = document.getElementById("bookmark-editor-rotation-y");
    this._rotationZDiv      = document.getElementById("bookmark-editor-rotation-z");
    this._rotationWDiv      = document.getElementById("bookmark-editor-rotation-w");

    // Connect buttons setting fields to current values --------------------------------------------

    document.querySelector("#bookmark-editor-start-date + div > button").onclick = () => {
      this._startDateDiv.value =
          CosmoScout.state.simulationTime.toISOString().replace('T', ' ').slice(0, 19);
    };

    document.querySelector("#bookmark-editor-end-date + div > button").onclick = () => {
      this._endDateDiv.value =
          CosmoScout.state.simulationTime.toISOString().replace('T', ' ').slice(0, 19);
    };

    document.querySelector("#bookmark-editor-frame + div > button").onclick = () => {
      this._centerDiv.value = CosmoScout.state.activePlanetCenter;
      this._frameDiv.value  = CosmoScout.state.activePlanetFrame;
    };

    document.querySelector("#bookmark-editor-location-z + div > button").onclick = () => {
      this._locationXDiv.value = CosmoScout.state.observerPosition[0];
      this._locationYDiv.value = CosmoScout.state.observerPosition[1];
      this._locationZDiv.value = CosmoScout.state.observerPosition[2];
    };

    document.querySelector("#bookmark-editor-rotation-w + div > button").onclick = () => {
      this._rotationXDiv.value = CosmoScout.state.observerRotation[0];
      this._rotationYDiv.value = CosmoScout.state.observerRotation[1];
      this._rotationZDiv.value = CosmoScout.state.observerRotation[2];
      this._rotationWDiv.value = CosmoScout.state.observerRotation[3];
    };

    // Delete bookmarks on delete button click -----------------------------------------------------

    this._deleteButton.onclick = () => {
      if (this._editBookmarkID != null) {
        CosmoScout.callbacks.bookmark.remove(this._editBookmarkID);
        this._editBookmarkID = null;
      }
      this._editor.classList.remove("visible");
    };

    // Save bookmark on save button click ----------------------------------------------------------

    this._saveButton.onclick = () => {
      // First validate all fields. We check for completeness first, then for correct formatting.
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

      let allCorrect = true;

      let markInvalid = (div, isInvalid, message) => {
        if (isInvalid) {
          div.classList.add("is-invalid");
          div.parentNode.querySelector(".invalid-feedback").textContent = message;
          allCorrect                                                    = false;
        } else {
          div.classList.remove("is-invalid");
        }
      };

      markInvalid(this._nameDiv, !nameGiven, "Please choose a name.");

      markInvalid(this._startDateDiv, endDateGiven && !startDateGiven,
          "You have to specify a start date if you set an end date.");

      markInvalid(this._centerDiv, !centerGiven && centerRequired, "Both values are required.");

      markInvalid(this._frameDiv, !frameGiven && frameRequired, "Both values are required.");

      markInvalid(this._locationXDiv, locationRequired && this._locationXDiv.value == "",
          "All values are required.");

      markInvalid(this._locationYDiv, locationRequired && this._locationYDiv.value == "",
          "All values are required.");

      markInvalid(this._locationZDiv, locationRequired && this._locationZDiv.value == "",
          "All values are required.");

      markInvalid(this._rotationXDiv, rotationRequired && this._rotationXDiv.value == "",
          "Provide all rotation values or none.");

      markInvalid(this._rotationYDiv, rotationRequired && this._rotationYDiv.value == "",
          "Provide all rotation values or none.");

      markInvalid(this._rotationZDiv, rotationRequired && this._rotationZDiv.value == "",
          "Provide all rotation values or none.");

      markInvalid(this._rotationWDiv, rotationRequired && this._rotationWDiv.value == "",
          "Provide all rotation values or none.");

      if (nameGiven && !(startDateGiven || endDateGiven || centerGiven || frameGiven ||
                           anyLocationGiven || anyRotationGiven)) {
        this._nothingGivenError.style.display = "block";
        allCorrect                            = false;
      } else {
        this._nothingGivenError.style.display = "none";
      }

      // Abort if errors occurred.
      if (!allCorrect) {
        return;
      }

      // Now check some formatting of the position, rotation and date inputs.

      // Based on https://stackoverflow.com/questions/3143070/javascript-regex-iso-datetime
      let dateRegex =
          /(^\d{4}-[01]\d-[0-3]\d [0-2]\d:[0-5]\d:[0-5]\d\.\d+$)|(^\d{4}-[01]\d-[0-3]\d [0-2]\d:[0-5]\d:[0-5]\d$)|(^\d{4}-[01]\d-[0-3]\d [0-2]\d:[0-5]\d$)/;
      let dateError =
          "Date must be in the format YYYY-MM-DD HH:MM:SS.fff with seconds and milliseconds being optional.";

      if (startDateGiven) {
        markInvalid(this._startDateDiv, !this._startDateDiv.value.match(dateRegex), dateError);
      }

      if (endDateGiven) {
        markInvalid(this._endDateDiv, !this._endDateDiv.value.match(dateRegex), dateError);
      }

      let numRegex = /^[-+]?[0-9]+(\.[0-9]*)?$/;
      let numError = "Must be a number.";

      if (anyLocationGiven) {
        markInvalid(this._locationXDiv, !this._locationXDiv.value.match(numRegex), numError);
        markInvalid(this._locationYDiv, !this._locationYDiv.value.match(numRegex), numError);
        markInvalid(this._locationZDiv, !this._locationZDiv.value.match(numRegex), numError);
      }

      if (anyRotationGiven) {
        markInvalid(this._rotationXDiv, !this._rotationXDiv.value.match(numRegex), numError);
        markInvalid(this._rotationYDiv, !this._rotationYDiv.value.match(numRegex), numError);
        markInvalid(this._rotationZDiv, !this._rotationZDiv.value.match(numRegex), numError);
        markInvalid(this._rotationWDiv, !this._rotationWDiv.value.match(numRegex), numError);
      }

      // Abort if errors occurred.
      if (!allCorrect) {
        return;
      }

      // Now create the bookmark!
      let bookmark = {name: this._nameDiv.value, description: this._descriptionDiv.value};

      if (centerGiven) {
        bookmark.location = {center: this._centerDiv.value, frame: this._frameDiv.value};

        if (anyLocationGiven) {
          bookmark.location.position = [
            parseFloat(this._locationXDiv.value), parseFloat(this._locationYDiv.value),
            parseFloat(this._locationZDiv.value)
          ];
        }

        if (anyRotationGiven) {
          bookmark.location.rotation = [
            parseFloat(this._rotationXDiv.value), parseFloat(this._rotationYDiv.value),
            parseFloat(this._rotationZDiv.value), parseFloat(this._rotationWDiv.value)
          ];
        }
      }

      if (startDateGiven) {
        bookmark.time = {start: this._startDateDiv.value};

        if (endDateGiven) {
          bookmark.time.end = this._endDateDiv.value;
        }
      }

      let color      = CP.HEX(this._colorDiv.value);
      bookmark.color = [color[0] / 255.0, color[1] / 255.0, color[2] / 255.0];

      // If we were editing a bookmar, remove this.
      if (this._editBookmarkID != null) {
        CosmoScout.callbacks.bookmark.remove(this._editBookmarkID);
        this._editBookmarkID = null;
      }

      // Then add the newly create / edited bookmar as a new one.
      CosmoScout.callbacks.bookmark.add(JSON.stringify(bookmark));

      this._editor.classList.remove("visible");
    };
  }

  /**
   * Shows the editor and fills all fields with the values of the given bookmark.
   *
   * @param visible {boolean}
   */
  editBookmark(bookmarkID, bookmarkJSON) {
    this._editor.classList.add("visible");
    this._deleteButton.classList.remove("hidden");
    this._editBookmarkID    = bookmarkID;
    this._title.textContent = "Edit Bookmark";

    this._resetFields();

    let bookmark = JSON.parse(bookmarkJSON);

    this._colorDiv.picker.value(
        bookmark.color[0] * 255, bookmark.color[1] * 255, bookmark.color[2] * 255, 1);

    this._nameDiv.value        = bookmark.name;
    this._descriptionDiv.value = bookmark.description;

    if (bookmark.time) {
      this._startDateDiv.value = bookmark.time.start;

      if (bookmark.time.end) {
        this._endDateDiv.value = bookmark.time.end;
      }
    }

    if (bookmark.location) {
      this._centerDiv.value = bookmark.location.center;
      this._frameDiv.value  = bookmark.location.frame;

      if (bookmark.location.position) {
        this._locationXDiv.value = bookmark.location.position[0];
        this._locationYDiv.value = bookmark.location.position[1];
        this._locationZDiv.value = bookmark.location.position[2];
      }

      if (bookmark.location.rotation) {
        this._rotationXDiv.value = bookmark.location.rotation[0];
        this._rotationYDiv.value = bookmark.location.rotation[1];
        this._rotationZDiv.value = bookmark.location.rotation[2];
        this._rotationWDiv.value = bookmark.location.rotation[3];
      }
    }
  }

  /**
   * Opens the editor and clears all fields to create a new bookmark.
   *
   * @param visible {boolean}
   */
  addNewBookmark() {
    this._editor.classList.add("visible");
    this._deleteButton.classList.add("hidden");
    this._editBookmarkID    = null;
    this._title.textContent = "Add New Bookmark";

    this._resetFields();
  }

  _resetFields() {
    this._colorDiv.picker.value(220, 170, 255, 1);
    this._nameDiv.value        = "";
    this._descriptionDiv.value = "";
    this._startDateDiv.value   = "";
    this._endDateDiv.value     = "";
    this._centerDiv.value      = "";
    this._frameDiv.value       = "";
    this._locationXDiv.value   = "";
    this._locationYDiv.value   = "";
    this._locationZDiv.value   = "";
    this._rotationXDiv.value   = "";
    this._rotationYDiv.value   = "";
    this._rotationZDiv.value   = "";
    this._rotationWDiv.value   = "";
  }
}
