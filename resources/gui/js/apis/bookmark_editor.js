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
  _iconButton;
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

  _bookmarkTooltipContainer;
  _bookmarkTooltipGotoLocation;
  _bookmarkTooltipGotoTime;
  _bookmarkTooltipEdit;
  _bookmarkTooltipName;
  _bookmarkTooltipDescription;
  _bookmarkTooltipArrow;

  _editBookmarkID = null;

  /**
   * @inheritDoc
   */
  init() {
    this._title             = document.getElementById("bookmark-editor-title");
    this._editor            = document.getElementById("bookmark-editor");
    this._nothingGivenError = document.getElementById("bookmark-editor-nothing-given-error");
    this._iconButton        = document.querySelector("#bookmark-editor-icon-select-button img");
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

    this._bookmarkTooltipContainer    = document.getElementById('bookmark-tooltip-container');
    this._bookmarkTooltipGotoLocation = document.getElementById('bookmark-tooltip-goto-location');
    this._bookmarkTooltipGotoTime     = document.getElementById('bookmark-tooltip-goto-time');
    this._bookmarkTooltipEdit         = document.getElementById('bookmark-tooltip-edit');
    this._bookmarkTooltipName         = document.getElementById('bookmark-tooltip-name');
    this._bookmarkTooltipDescription  = document.getElementById('bookmark-tooltip-description');
    this._bookmarkTooltipArrow        = document.getElementById('bookmark-tooltip-arrow');

    // Make sure that the tooltip is hidden.
    this._bookmarkTooltipContainer.onmouseout = () => {
      this.hideBookmarkTooltip();
    };

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

    // Initialize Icon Select Popover --------------------------------------------------------------

    $("#bookmark-editor-icon-select-button").on("shown.bs.popover", () => {
      let buttons = document.querySelectorAll("#bookmark-editor-icon-select-list a");
      buttons.forEach((b) => {
        b.onclick = () => {
          if (b.children.length > 0) {
            this.selectIcon(b.children[0].getAttribute("src"));
          } else {
            this.selectIcon("");
          }
          $("#bookmark-editor-icon-select-button").popover("hide");
        };
      });
    });

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

      var textRegex = /[\\"<>]/;
      let textError = "This shouldn't contain special characters like \\, \", < or >.";
      markInvalid(this._nameDiv, this._nameDiv.value.match(textRegex), textError);
      markInvalid(this._descriptionDiv, this._descriptionDiv.value.match(textRegex), textError);

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

      // Remove leading "../icons/" for the icon files.
      bookmark.icon = this._iconButton.getAttribute("src").slice(9);

      // If we were editing a bookmark, remove this.
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
   * Adds a possible icon to the icon select popover.
   *
   * @param path {string}
   */
  addIcon(path) {
    document.getElementById("bookmark-editor-icon-select-list").innerHTML +=
        `<div class="col-3 p-1">
          <a class="btn block glass">
            <img class="img-fluid" src="../icons/${path}">
          </a>
        </div>`;
  }

  /**
   * Selects an icon for the current bookmark.
   *
   * @param path {string}
   */
  selectIcon(path) {
    document.querySelector("#bookmark-editor-icon-select-button img").setAttribute("src", path);
  }

  /**
   * Shows the editor and fills all fields with the values of the given bookmark.
   *
   * @param bookmarkID {number}
   * @param bookmarkJSON {string}
   */
  editBookmark(bookmarkID, bookmarkJSON) {
    this._editor.classList.add("visible");
    this._deleteButton.classList.remove("hidden");
    this._editBookmarkID    = bookmarkID;
    this._title.textContent = "Edit Bookmark";

    this._resetFields();

    let bookmark = JSON.parse(bookmarkJSON);

    if (bookmark.color) {
      this._colorDiv.picker.value(
          bookmark.color[0] * 255, bookmark.color[1] * 255, bookmark.color[2] * 255, 1);
    }

    if (bookmark.icon) {
      this._iconButton.setAttribute("src", "../icons/" + bookmark.icon);
    }

    this._nameDiv.value = bookmark.name;

    if (bookmark.description) {
      this._descriptionDiv.value = bookmark.description;
    }

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
   */
  addNewBookmark() {
    this._editor.classList.add("visible");
    this._deleteButton.classList.add("hidden");
    this._editBookmarkID    = null;
    this._title.textContent = "Add New Bookmark";

    this._resetFields();
  }

  /**
   * Use this to show a bookmark tooltip somewhere. This is, for example, used by the timeline.
   *
   * @param id              {number}
   * @param name            {string}
   * @param description     {string}
   * @param hasLocation     {boolean}
   * @param hasTime         {boolean}
   * @param tooltipPosition {Array}
   */
  showBookmarkTooltip(id, name, description, hasLocation, hasTime, tooltipX, tooltipY) {

    // Show the tooltip.
    this._bookmarkTooltipContainer.classList.add('visible');

    // Fill all the fields.
    if (hasLocation) {
      this._bookmarkTooltipGotoLocation.classList.remove('hidden');
      this._bookmarkTooltipGotoLocation.onclick = () => {
        CosmoScout.callbacks.bookmark.gotoLocation(id);
      };
    } else {
      this._bookmarkTooltipGotoLocation.classList.add('hidden');
    }

    if (hasTime) {
      this._bookmarkTooltipGotoTime.classList.remove('hidden');
      this._bookmarkTooltipGotoTime.onclick = () => {
        CosmoScout.callbacks.bookmark.gotoTime(id, 2.0);
      };
    } else {
      this._bookmarkTooltipGotoTime.classList.add('hidden');
    }

    this._bookmarkTooltipEdit.onclick = () => {
      CosmoScout.callbacks.bookmark.edit(id);
    };

    this._bookmarkTooltipName.innerHTML        = name;
    this._bookmarkTooltipDescription.innerHTML = description;

    // Calculate a position.
    const tooltipWidth  = this._bookmarkTooltipContainer.offsetWidth;
    const tooltipHeight = this._bookmarkTooltipContainer.offsetHeight;
    const arrowWidth    = 10;
    const tooltipOffset = 20;

    const left = Math.max(
        0, Math.min(document.body.offsetWidth - tooltipWidth, tooltipX - tooltipWidth / 2));
    this._bookmarkTooltipArrow.style.left     = `${tooltipX - left - arrowWidth}px`;
    this._bookmarkTooltipContainer.style.left = `${left}px`;

    if (tooltipHeight + tooltipOffset < tooltipY) {
      // Position above.
      this._bookmarkTooltipContainer.style.top = `${tooltipY - tooltipOffset - tooltipHeight}px`;
      this._bookmarkTooltipArrow.classList     = ["top"];
    } else {
      // Position below.
      this._bookmarkTooltipContainer.style.top = `${tooltipY + tooltipOffset}px`;
      this._bookmarkTooltipArrow.classList     = ["bottom"];
    }
  }

  // The tooltip will be hidden when the pointer leaves it, but it may stay open indefinitely if the
  // user never hovers it. So you should call this if once the bookmark for which the tooltip is
  // shown is not hovered anymore.
  hideBookmarkTooltip() {
    this._bookmarkTooltipContainer.classList.remove('visible');
  }

  _resetFields() {
    this._colorDiv.picker.value(220, 170, 255, 1);
    this._iconButton.setAttribute("src", "");
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
