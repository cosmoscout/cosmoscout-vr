/* global IApi, Format, CosmoScout */

/**
 * Statusbar Api
 */
class StatusbarApi extends IApi {
  /**
   * @inheritDoc
   */
  name = 'statusbar';

  /**
   * @type {HTMLElement}
   * @private
   */
  _userContainer;

  /**
   * @type {HTMLElement}
   * @private
   */
  _pointerContainer;

  /**
   * @type {HTMLElement}
   * @private
   */
  _speedContainer;

  _history = [];
  _historyIndex = 0;
  _currentCmd = "";
  _inputField;
  _suggestionField;
  _outputField;

  /**
   * Initialize all containers
   */
  init() {
    this._userContainer = document.querySelector("#statusbar-user-position");
    this._pointerContainer = document.querySelector("#statusbar-pointer-position");
    this._speedContainer = document.querySelector("#statusbar-speed");
    this._inputField = document.querySelector("#console-input-area input");
    this._suggestionField = document.querySelector("#console-suggestion-area");
    this._outputField = document.querySelector("#console-output-area");

    let self = this;

    this._inputField.addEventListener('keydown', function (e) {
      // Up pressed - history up
      if (e.keyCode == 38) {
        if (self._history.length > 0) {
          if (self._historyIndex == self._history.length) {
            this._currentCmd = self._inputField.value;
          }
          self._historyIndex = Math.max(0, self._historyIndex - 1);
          self._inputField.value = self._history[self._historyIndex];
        }
        e.preventDefault();
      }

      // Down pressed - history down
      if (e.keyCode == 40) {
        if (self._history.length > 0) {
          self._historyIndex = Math.min(self._history.length, self._historyIndex + 1);
          if (self._historyIndex == self._history.length) {
            self._inputField.value = this._currentCmd;
          } else {
            self._inputField.value = self._history[self._historyIndex];
          }
        }
        e.preventDefault();
      }
    });

    this._inputField.addEventListener('keypress', function (e) {

      self._enableSuggestionArea(false);

      // Return pressed - try to execute the command!
      if (e.keyCode == 13) {
        try {
          let result = eval(self._inputField.value);
          if (result != undefined) {
            console.log(result);
          }
        } catch (error) {
          console.warn(error);
        }

        if (self._history.length == 0 || self._history[self._history.length - 1] != self._inputField.value) {
          self._history.push(self._inputField.value);
        }

        self._historyIndex = self._history.length;
        self._inputField.value = ""
      }

      // Tab pressed - auto complete
      if (e.keyCode == 9) {
        e.preventDefault();

        let cursorPos = self._inputField.selectionStart;
        let text = self._inputField.value.substring(0, cursorPos);

        let objectEnd = text.lastIndexOf(".");
        let objectBegin = 0;

        // find last occurrence of " " , ; + - * / ( ) { } | & !
        let regex = new RegExp("\\s|,|;|\\+|-|\\*|/|\\(|\\)|{|}|\\||&|\\!", "g");
        let match;
        while ((match = regex.exec(text)) != null) {
          objectBegin = match.index + 1;
        }

        let objectName = "window";
        let prefixBegin = 0;

        if (objectEnd < objectBegin) {
          prefixBegin = objectBegin;
        } else if (objectEnd > 0 && objectBegin < cursorPos - 1) {
          objectName = text.substring(Math.max(0, objectBegin), objectEnd);
          prefixBegin = objectEnd + 1;
        }

        let prefix = text.substring(prefixBegin);

        let object = eval(objectName);

        if (object != undefined) {

          let properties = Object.getOwnPropertyNames(object);
          let proto = Object.getPrototypeOf(object);

          if (proto && proto != Object.prototype && typeof proto !== "function") {
            properties = properties.concat(Object.getOwnPropertyNames(proto))
          }

          properties = properties.filter(element => prefix === "" || element.startsWith(prefix)).sort();

          let prefixEnd = cursorPos;

          // find next occurrence of " " , ; + - * / ( ) { } | & ! [ ]
          let regex = new RegExp("\\s|,|;|\\+|-|\\*|/|\\(|\\)|{|}|\\||&|\\!|\\[|\\]", "g");
          regex.lastIndex = cursorPos;
          match = regex.exec(self._inputField.value);

          if (match != null) {
            prefixEnd = match.index;
          }

          let getCompletion = (element) => {
            let completion = element;
            let finalCursorPos = prefixBegin + completion.length;
            if (typeof object[completion] === "function") {
              completion += "()";
              finalCursorPos += 1;
            }

            if (typeof object[completion] === "object") {
              completion += ".";
              finalCursorPos += 1;
            }

            return [completion, finalCursorPos];
          }

          if (properties.length == 1) {
            let [completion, finalCursorPos] = getCompletion(properties[0]);
            self._setCompletion(prefixBegin, prefixEnd, finalCursorPos, completion);
          } else {
            self._suggestionField.innerHTML = "";
            properties.forEach(element => {
              let [completion, finalCursorPos] = getCompletion(element);
              let classNames = `suggestion type-${typeof object[element]}`;

              if (completion.startsWith("_")) {
                classNames += " private";
              }

              self._suggestionField.insertAdjacentHTML("beforeend",
                `<span class='${classNames}'
                       onclick='CosmoScout.statusbar._setCompletion(${prefixBegin}, ${prefixEnd}, 
                                                              ${finalCursorPos}, "${completion}");'>
                       ${element}
                </span>`);
              self._enableSuggestionArea(true);
            });
          }
        }
      }
    }, true);
  }

  update() {
    let pos = CosmoScout.state.pointerPosition;
    if (pos !== undefined) {
      this._pointerContainer.innerText = `${CosmoScout.utils.formatLongitude(pos[0]) + CosmoScout.utils.formatLatitude(pos[1])}(${CosmoScout.utils.formatHeight(pos[2])})`;
    } else {
      this._pointerContainer.innerText = ' - ';
    }

    pos = CosmoScout.state.observerPosition;
    if (pos !== undefined) {
      this._userContainer.innerText = `${CosmoScout.utils.formatLongitude(pos[0]) + CosmoScout.utils.formatLatitude(pos[1])}(${CosmoScout.utils.formatHeight(pos[2])})`;
    } else {
      this._userContainer.innerText = ' - ';
    }

    if (CosmoScout.state.observerSpeed !== undefined) {
      this._speedContainer.innerText = CosmoScout.utils.formatSpeed(CosmoScout.state.observerSpeed);
    }
  }

  printMessage(level, channel, message) {
    this._outputField.insertAdjacentHTML("afterbegin", `<div class='message level-${level}'>
                                                    [${level}] ${channel} ${message}
                                                  </div>`);

    while (this._outputField.children.length > 100) {
      this._outputField.removeChild(this._outputField.lastChild);
    }

    // Flush all pending style changes to ensure the initial transition gets triggered
    getComputedStyle(this._outputField.firstChild).opacity;
    this._outputField.firstChild.classList.add("initial-transition");
  }

  _enableSuggestionArea(enable) {
    if (enable) {
      this._suggestionField.classList.add("show");
    } else {
      this._suggestionField.classList.remove("show");
    }
  }

  _setCompletion(startIndex, endIndex, finalCursorPos, text) {
    this._inputField.value = this._inputField.value.substring(0, startIndex)
      + text
      + this._inputField.value.substring(endIndex);
    this._inputField.setSelectionRange(finalCursorPos, finalCursorPos);
    this._inputField.focus();
    this._enableSuggestionArea(false);
  }
}
