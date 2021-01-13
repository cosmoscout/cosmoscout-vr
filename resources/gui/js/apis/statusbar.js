/* global IApi, Format, CosmoScout */

/**
 * This is a default CosmoScout API. Once initialized, you can access its methods via
 * CosmoScout.statusbar.<method name>. The only public method you may want to call is
 * CosmoScout.statusbar.printMessage(), which will print something to the console.
 * However, this is not recommended as all messages logged with console.log() will be
 * printed to the console anyways.
 */
class StatusbarApi extends IApi {
  /**
   * @inheritDoc
   */
  name = 'statusbar';

  /**
   * The list of previously executed commands.
   */
  history = [];

  /**
   * The currently selected item in the history.
   */
  historyIndex = 0;

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

  _currentCmd = "";
  _inputField;
  _suggestionField;
  _outputField;
  _outputWrapper;

  /**
   * Initializes the statusbar and the on-screen console.
   */
  init() {

    // Store required containers for faster access.
    this._userContainer    = document.querySelector("#statusbar-user-position");
    this._pointerContainer = document.querySelector("#statusbar-pointer-position");
    this._speedContainer   = document.querySelector("#statusbar-speed");
    this._inputField       = document.querySelector("#console-input-area input");
    this._suggestionField  = document.querySelector("#console-suggestion-area");
    this._outputField      = document.querySelector("#console-output-area");
    this._outputWrapper    = document.querySelector("#console-output-wrapper");

    // The 'console-has-input-focus' class on the _outputWrapper forces the console messages to not
    // fade when the text input field has input focus.
    this._inputField.onfocus = (e) => {
      this._outputWrapper.classList.add('console-has-input-focus');
    };

    this._inputField.onblur = (e) => {
      this._outputWrapper.classList.remove('console-has-input-focus');
    };

    this._inputField.onkeydown = (e) => {
      // Up pressed - history up.
      if (e.keyCode == 38) {
        if (this.history.length > 0) {
          if (this.historyIndex == this.history.length) {
            this._currentCmd = this._inputField.value;
          }
          this.historyIndex      = Math.max(0, this.historyIndex - 1);
          this._inputField.value = this.history[this.historyIndex];
        }
        e.preventDefault();
      }

      // Down pressed - history down.
      if (e.keyCode == 40) {
        if (this.history.length > 0) {
          this.historyIndex = Math.min(this.history.length, this.historyIndex + 1);
          if (this.historyIndex == this.history.length) {
            this._inputField.value = this._currentCmd;
          } else {
            this._inputField.value = this.history[this.historyIndex];
          }
        }
        e.preventDefault();
      }
    };

    this._inputField.onkeypress = (e) => {
      this._enableSuggestionArea(false);

      // Return pressed - try to execute the command!
      if (e.keyCode == 13) {
        try {
          let result = window.eval(this._inputField.value);
          if (result != undefined) {
            console.log(result);
          } else {
            console.log(this._inputField.value);
          }
        } catch (error) { console.warn(error); }

        // Push command to history.
        if (this.history.length == 0 ||
            this.history[this.history.length - 1] != this._inputField.value) {
          this.history.push(this._inputField.value);
          CosmoScout.callbacks.statusbar.addCommandToHistory(this._inputField.value);
        }

        this.historyIndex      = this.history.length;
        this._inputField.value = ""
      }

      // Tab pressed - auto complete.
      if (e.keyCode == 9) {
        e.preventDefault();

        // Store position of cursor for better readability.
        let cursorPos = this._inputField.selectionStart;

        // Get current command until position of cursor.
        let text = this._inputField.value.substring(0, cursorPos);

        // We will suggest properties of object directly preceding the cursor. The name of the
        // object has to end with a '.', the start of the object can be any of the characters in
        // the regex below.
        let objectEnd   = text.lastIndexOf(".");
        let objectBegin = 0;

        // find last occurrence of " " , ; + - * / ( ) { } | & !
        let regex = new RegExp("\\s|,|;|\\+|-|\\*|/|\\(|\\)|{|}|\\||&|\\!", "g");
        let match;
        while ((match = regex.exec(text)) != null) {
          objectBegin = match.index + 1;
        }

        // Now we have to get the object's name and the prefix of the property which is to
        // completed. If there is no '.' preceding the cursor, there is no object and we have to
        // look for global variable (object name = 'window'). The prefix is everything between
        // the '.' and the cursor position (or the entire command if there is no '.').
        let prefixBegin = 0;
        let objectName  = "window";

        if (objectEnd < objectBegin) {
          prefixBegin = objectBegin;
        } else if (objectEnd > 0 && objectBegin < cursorPos - 1) {
          objectName  = text.substring(Math.max(0, objectBegin), objectEnd);
          prefixBegin = objectEnd + 1;
        }

        let prefix = text.substring(prefixBegin);

        // Now that we have the object's name, we can get the object by evaluating it.
        let object = eval(objectName);

        // Now we can loop through all properties of the object and find suitable
        // completion candidates.
        if (object != undefined) {

          // We suggest all properties of the object and it's __proto__ part.
          let properties = Object.getOwnPropertyNames(object);
          let proto      = Object.getPrototypeOf(object);

          if (proto && proto != Object.prototype && typeof proto !== "function") {
            properties = properties.concat(Object.getOwnPropertyNames(proto))
          }

          // Now we filter the list to contain only those with our prefix and sort it
          // alphabetically.
          properties =
              properties.filter(element => prefix === "" || element.startsWith(prefix)).sort();

          // If the cursor is somewhere in the middle of a property name, we want to replace the
          // entire property with our completion. We use another regex to find the end of the text
          // we want to replace.
          let prefixEnd = cursorPos;

          // Find next occurrence of " " , ; + - * / ( ) { } | & ! [ ]
          let regex       = new RegExp("\\s|,|;|\\+|-|\\*|/|\\(|\\)|{|}|\\||&|\\!|\\[|\\]", "g");
          regex.lastIndex = cursorPos;
          match           = regex.exec(this._inputField.value);

          if (match != null) {
            prefixEnd = match.index;
          }

          // If the thing we suggest for completion is an object, we append a '.', if it's a
          // function we add '()' and place the cursor between the brackets.
          let getCompletion = (element) => {
            let completion     = element;
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
          };

          if (properties.length == 1) {
            // If there is only one possible completion, we directly apply it.
            let [completion, finalCursorPos] = getCompletion(properties[0]);
            this._setCompletion(prefixBegin, prefixEnd, finalCursorPos, completion);

          } else if (properties.length > 1) {
            // If there are multiple completion possibilities, we show a list and complete as much
            // as possible (the longest prefix shared by all suggestions).
            this._suggestionField.innerHTML = "";

            // This will be truncated to the longest shared prefix.
            let maximumCompletion = properties[0];

            properties.forEach(element => {
              let [completion, finalCursorPos] = getCompletion(element);
              let classNames                   = `suggestion type-${typeof object[element]}`;

              // Compare this completion candidate with our current longest shared prefix. Truncate
              // this if needed.
              let i = 0;
              while (i < completion.length && i < maximumCompletion.length &&
                     completion.charAt(i) === maximumCompletion.charAt(i)) {
                i++;
              }
              maximumCompletion = maximumCompletion.substring(0, i);

              // If it's a "private" property, add a class name. These items will be drawn less
              // opaque.
              if (completion.startsWith("_")) {
                classNames += " private";
              }

              // Finally add the item to the list of completions. Clicking it will apply the
              // completion.
              this._suggestionField.insertAdjacentHTML("beforeend", `<span class='${classNames}'
                       onclick='CosmoScout.statusbar._setCompletion(${prefixBegin}, ${prefixEnd}, 
                                                              ${finalCursorPos}, "${completion}");'>
                       ${element}
                </span>`);
            });

            // Set the longest shared prefix as completion and place the cursor to the end.
            this._setCompletion(
                prefixBegin, prefixEnd, prefixBegin + maximumCompletion.length, maximumCompletion);

            // Finally show the completion area.
            this._enableSuggestionArea(true);
          }
        }
      }
    };
  }

  /**
   * This is called once a frame and updates all status div's of the statusbar.
   */
  update() {
    let pos = CosmoScout.state.pointerPosition;
    if (pos !== undefined) {
      this._pointerContainer.innerText =
          `${CosmoScout.utils.formatLongitude(pos[0]) + CosmoScout.utils.formatLatitude(pos[1])}(${
              CosmoScout.utils.formatHeight(pos[2])})`;
    } else {
      this._pointerContainer.innerText = ' - ';
    }

    pos = CosmoScout.state.observerLngLatHeight;
    if (pos !== undefined) {
      this._userContainer.innerText =
          `${CosmoScout.utils.formatLongitude(pos[0]) + CosmoScout.utils.formatLatitude(pos[1])}(${
              CosmoScout.utils.formatHeight(pos[2])})`;
    } else {
      this._userContainer.innerText = ' - ';
    }

    if (CosmoScout.state.observerSpeed !== undefined) {
      this._speedContainer.innerText = CosmoScout.utils.formatSpeed(CosmoScout.state.observerSpeed);
    }
  }

  /**
   * Print a message to the console.
   * @param level   {string} This should be either "T", "D", "I", "W", "E" or "C".
   * @param channel {string} This should usually be the name of the logger.
   * @param message {string} The message.
   */
  printMessage(level, channel, message) {
    this._outputField.insertAdjacentHTML(
        "afterbegin", `<div class='message level-${level}'>[${level}] ${channel} ${message}</div>`);

    while (this._outputField.children.length > 100) {
      this._outputField.removeChild(this._outputField.lastChild);
    }

    // Flush all pending style changes to ensure the initial transition gets triggered
    getComputedStyle(this._outputField.firstChild).opacity;
    this._outputField.firstChild.classList.add("initial-transition");
  }

  /**
   * Toggles the visibility of the suggestion area.
   * @param {bool} enable
   */
  _enableSuggestionArea(enable) {
    if (enable) {
      this._suggestionField.classList.add("show");
    } else {
      this._suggestionField.classList.remove("show");
    }
  }

  /**
   * Adds some text at a specific position into the current command in the console input area.
   * @param {number} startIndex     Index where the text should be inserted
   * @param {number} endIndex       Index until which the existing text will be overwritten
   * @param {number} finalCursorPos The cursor will be placed at this index
   * @param {number} text           The text to insert.
   */
  _setCompletion(startIndex, endIndex, finalCursorPos, text) {
    this._inputField.value = this._inputField.value.substring(0, startIndex) + text +
                             this._inputField.value.substring(endIndex);
    this._inputField.setSelectionRange(finalCursorPos, finalCursorPos);
    this._inputField.focus();
    this._enableSuggestionArea(false);
  }
}
