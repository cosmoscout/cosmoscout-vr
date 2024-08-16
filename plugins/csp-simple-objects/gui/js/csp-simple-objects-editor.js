////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2022 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

(() => {

    class SimpleObjectsEditorApi extends IApi {
        /**
         * @inheritDoc
         */
        name = 'simpleObjectsEditor';
       
        // Buttons
        _pickLocationButton;
        _saveButton;
        _deleteButton;
        
        // Input fields
        _nameDiv;
        _anchorDiv;
        _modelSelect;
        _mapSelect;
        _modelButton;
        _mapButton;
        _locationLngDiv;
        _locationLatDiv;
        _elevationDiv;
        _scaleDiv;
        _sizeDiv;
        _rotationXDiv;
        _rotationYDiv;
        _rotationZDiv;
        _rotationWDiv;
        _alignSurfaceDiv;
        _nothingGivenError;
        
        // Window variables
        _editor;
        _title;
        _editorOpen = false;
        _editObjectName = null;
        _objectNames = new Set();       // contains all known object names
        
        // Pick location related variables
        _lastHeight = Infinity;
        _pickLocationButtonActive = false;
        
        // Tab functionality maps 
        _tabFwdMap = new Map();
        //tabRevMap = new Map();

        /**
         * @inheritDoc
         */
        init() {
            this._title = document.getElementById("simple-objects-editor-title");
            this._editor = document.getElementById("simple-objects-editor");
            this._nothingGivenError = document.getElementById("simple-objects-editor-nothing-given-error");
            this._pickLocationButton = document.getElementById("simple-objects-editor-pick-location");
            this._saveButton = document.getElementById("simple-objects-editor-save-button");
            this._deleteButton = document.getElementById("simple-objects-editor-delete-button");
            this._nameDiv = document.getElementById("simple-objects-editor-name");
            this._anchorDiv = document.getElementById("simple-objects-editor-anchor");
            this._modelSelect = document.getElementById("simple-objects-editor-model-file");
            this._mapSelect = document.getElementById("simple-objects-editor-map-file");
            this._locationLngDiv = document.getElementById("simple-objects-editor-location-lng");
            this._locationLatDiv = document.getElementById("simple-objects-editor-location-lat");
            this._elevationDiv = document.getElementById("simple-objects-editor-elevation");
            this._scaleDiv = document.getElementById("simple-objects-editor-scale");
            this._sizeDiv = document.getElementById("simple-objects-editor-size");
            this._rotationXDiv = document.getElementById("simple-objects-editor-rotation-x");
            this._rotationYDiv = document.getElementById("simple-objects-editor-rotation-y");
            this._rotationZDiv = document.getElementById("simple-objects-editor-rotation-z");
            this._rotationWDiv = document.getElementById("simple-objects-editor-rotation-w");
            this._alignSurfaceDiv = document.getElementById("simple-objects-editor-align-surface");

            /**
             * Is only used for the dropdown buttons that need to be initialized in advance. 
             * The initalization adds the inner button HTMLElement after which a div with 
             * the invalid-feedback class needs to be added to display a false input correctly.
             * @param {HTMLElement} div Button HTMLElement after which the invalid-feedback div should be added to.
             */
            let addInvalidFeedback = (div) => {
                if (div.parentNode.querySelector(".invalid-feedback") == null) {
                    let el = document.createElement("div");
                    el.classList.add("invalid-feedback");
                    div.after(el);
                }
            }

            // Dropdown needs to be initialized before invalid-feedback class is added to the button divs
            CosmoScout.gui.initDropDowns();

            this._modelButton = document.querySelector("#simple-objects-editor-model-file + button");
            this._mapButton = document.querySelector("#simple-objects-editor-map-file + button");
            this._modelButton.classList.add("form-control");
            this._mapButton.classList.add("form-control");
            
            addInvalidFeedback(this._modelButton);
            addInvalidFeedback(this._mapButton);


            /**
             * Tab loop initilization.
             * Only forward tabbing works (see below).
             * 
             * 
             * Funktionalität für rückwärts tabben durch prevDiv und this.tabRevMap,
             * aber rückwärts funktioniert nicht..  :(
             * 
             * Weil: "shift+tab" triggert keypress event nicht.
             *  ->  nur "tab" (keyCodwe == 9) triggert keypress event. 
             *  
             * keydown wird durch "shift" getriggert, aber nicht durch "tab"?!
             * -> während shift gedrückt ist kommt auch kein keypress event von tab. WARUM?!
             * 
             * Ist das irgendwo im code deaktiviert oder ist das hier eine ALTE VERSION?
             * @param {HTMLElement} div 
             * @param {HTMLElement} prevDiv 
             * @param {HTMLElement} nextDiv 
             */
            let initTabCallback = (div, prevDiv, nextDiv) => {
                //this.tabRevMap.set(div, prevDiv);
                this._tabFwdMap.set(div, nextDiv);

                div.onkeypress = (e) => {
                    if (e.keyCode == 9) {
                        e.preventDefault();
                        // let nextElement = null;
                        // if(e.shiftKey) {
                        //     nextElement = this.tabRevMap.get(document.activeElement)
                        // } else {
                        let nextElement = this._tabFwdMap.get(document.activeElement)      
                        //}
                        if (nextElement != null) {
                            nextElement.focus();
                        }
                    }
                };
            }

            initTabCallback(this._nameDiv, this._rotationWDiv, this._anchorDiv);
            initTabCallback(this._anchorDiv, this._nameDiv, this._locationLngDiv);
            initTabCallback(this._locationLngDiv, this._anchorDiv, this._locationLatDiv);
            initTabCallback(this._locationLatDiv, this._locationLngDiv, this._elevationDiv);
            initTabCallback(this._elevationDiv, this._locationLatDiv, this._scaleDiv);
            initTabCallback(this._scaleDiv, this._elevationDiv, this._sizeDiv);
            initTabCallback(this._sizeDiv, this._scaleDiv, this._rotationXDiv);
            initTabCallback(this._rotationXDiv, this._scaleDiv, this._rotationYDiv);
            initTabCallback(this._rotationYDiv, this._rotationXDiv, this._rotationZDiv);
            initTabCallback(this._rotationZDiv, this._rotationYDiv, this._rotationWDiv);
            initTabCallback(this._rotationWDiv, this._rotationZDiv, this._nameDiv);

            document.querySelector("#simple-objects-editor-anchor + div > button").onclick = () => {
                this._anchorDiv.value = CosmoScout.state.activePlanetCenter;
            };

            document.querySelector("#simple-objects-editor-rotation-w + div > button").onclick = () => {
                this._rotationXDiv.value = CosmoScout.state.observerRotation[0];
                this._rotationYDiv.value = CosmoScout.state.observerRotation[1];
                this._rotationZDiv.value = CosmoScout.state.observerRotation[2];
                this._rotationWDiv.value = CosmoScout.state.observerRotation[3];
            };
            
            document.querySelector("#simple-objects-editor-pick-location").onclick = () => {
                this.setPickLocationEnabled(!this._pickLocationButtonActive);
            };

        // close button -------------------------------------------------------------------------------
            document.querySelector("#simple-objects-editor-close").onclick = () => {
                if(this._nameDiv.value != null) {
                    CosmoScout.callbacks.simpleObjects.undoEdit(this._nameDiv.value);
                }
                this._closeEditor();
            };

        // Delete bookmarks on delete button click -----------------------------------------------------
            this._deleteButton.onclick = () => {
                if (this._editObjectName != null && this.hasName(this._editObjectName)) {
                    this.remove(this._editObjectName);
                    this._editObjectName = null;
                }

                this._closeEditor();
            };

        // Save bookmark on save button click ----------------------------------------------------------
            this._saveButton.onclick = () => {

                if(!this.validateInput()) return;

                CosmoScout.callbacks.simpleObjects.save(this._editObjectName == null ? "" : this._editObjectName, 
                                                        this._nameDiv.value, JSON.stringify(this.generateJSONObject()));
                this._closeEditor();
            };

            /**
             * Adds a event listener to the given element, that invokes a timer after 
             * @param {HTMLElement} div 
             * @param {string} eventType 
             * @param {number} tdelay 
             */
            let initUpdateOnChange = (div, eventType = "keypress", tdelay = 750) => {
                let timeout = null;

                div.addEventListener(eventType, (e) => {
                    clearTimeout(timeout);
                    timeout = setTimeout(() => {
                        if(this.validateInput(true)) { this.updateModel(); }
                    }, tdelay);
                });
            }

            initUpdateOnChange(this._nameDiv);
            initUpdateOnChange(this._anchorDiv);
            initUpdateOnChange(this._modelSelect, "change");
            initUpdateOnChange(this._mapSelect, "change");
            initUpdateOnChange(this._locationLngDiv, "change");
            initUpdateOnChange(this._locationLatDiv, "change");
            initUpdateOnChange(this._locationLngDiv);
            initUpdateOnChange(this._locationLatDiv);
            initUpdateOnChange(this._elevationDiv);
            initUpdateOnChange(this._scaleDiv);
            initUpdateOnChange(this._sizeDiv);
            initUpdateOnChange(this._rotationXDiv);
            initUpdateOnChange(this._rotationYDiv);
            initUpdateOnChange(this._rotationZDiv);
            initUpdateOnChange(this._rotationWDiv);
            initUpdateOnChange(this._alignSurfaceDiv, "change", 0);

            initUpdateOnChange(this._nameDiv, "keyup");
            initUpdateOnChange(this._anchorDiv, "keyup");
            initUpdateOnChange(this._locationLngDiv, "keyup");
            initUpdateOnChange(this._locationLatDiv, "keyup");
            initUpdateOnChange(this._elevationDiv, "keyup");
            initUpdateOnChange(this._scaleDiv, "keyup");
            initUpdateOnChange(this._sizeDiv, "keyup");
            initUpdateOnChange(this._rotationXDiv, "keyup");
            initUpdateOnChange(this._rotationYDiv, "keyup");
            initUpdateOnChange(this._rotationZDiv, "keyup");
            initUpdateOnChange(this._rotationWDiv, "keyup");
        }


// --------------- basic functions used by the GUI --------------

        /**
         * Opens the editor and clears all fields to create a new object.
         */
        add() {
            this._resetFields();
            this._resetInvalid();
            this._deleteButton.classList.add("hidden");
            this._editObjectName = null;

            this._openEditor("Add new model object");
            this._nameDiv.focus();
        }
       
        /**
         * Opens the editor and parses the json parameters into their related fields.
         * It is called from the backend after the user clicked an edit button and 
         * the backend found the json config of the object.
         * @param {string} objectName 
         * @param {string} json 
         */
        edit(objectName, json) {
            this._resetFields();
            this._resetInvalid();

            this._deleteButton.classList.remove("hidden");

            let object = JSON.parse(json);

            this._editObjectName    = objectName;
            this._nameDiv.value     = objectName;
                        
            let modelFile           = object.modelFile.replace("../share/resources/models/","");
            let environmentMap      = object.environmentMap.replace("../share/resources/textures/","");
            
            this._modelSelect.value = modelFile;
            this._mapSelect.value   = environmentMap;
            this._modelButton.firstChild.firstChild.firstChild.innerHTML    = modelFile;
            this._mapButton.firstChild.firstChild.firstChild.innerHTML      = environmentMap;

            this._anchorDiv.value      = object.anchor;
            this._locationLngDiv.value = object.lngLat[0];
            this._locationLatDiv.value = object.lngLat[1];

            if(object.elevation)        { this._elevationDiv.value = object.elevation; }
            if(object.scale)            { this._scaleDiv.value     = object.scale; }
            if(object.diagonalLength)   { this._sizeDiv.value      = object.diagonalLength; }

            if(object.rotation) {
                this._rotationXDiv.value = object.rotation[0];
                this._rotationYDiv.value = object.rotation[1];
                this._rotationZDiv.value = object.rotation[2];
                this._rotationWDiv.value = object.rotation[3];
            }

            if(object.alignToSurface) {
                this._alignSurfaceDiv.checked = object.alignToSurface;
            }

            this._openEditor("Edit object");
        }


        /**
         * Removes the given object from the list and calls the backend remove function 
         * to erase the object (with the same name) and temporary object from the scene graph.
         * @param {string} objectName 
         */
        remove(objectName) {
            this.removeObjectFromList(objectName);
            CosmoScout.callbacks.simpleObjects.remove(this._editObjectName);
        }


        /**
         * Sends the current object configuration to the backend wich updates the temporary displayed object.
         * @returns If at least one mandatory parameter is not provided.
         */
        updateModel() {
            let nameGiven = this._nameDiv.value != "";
            let anchorGiven = this._anchorDiv.value != "";
            let modelGiven = !(this._modelSelect.value == "-1" || this._modelSelect.value == "");
            let mapGiven = !(this._mapSelect.value == "-1" || this._mapSelect.value == "");
            let locationGiven = this._locationLngDiv.value != "" && this._locationLngDiv.value != "";

            if(!(nameGiven && anchorGiven && modelGiven && mapGiven && locationGiven)) {
                return;
            }

            CosmoScout.callbacks.simpleObjects.update(this._nameDiv.value, JSON.stringify(this.generateJSONObject()));
 
            //console.debug("Updated Model");
        }
        
// ------------ Object list functions ---------------------

        /**
         * Adds a new item to the list of objects.
         * @param {string} objectName   Unformatted name of the object
         * @returns                     If the object already exists
         */
        addObjectToList(objectName) {

            if(this.hasName(objectName)) {
                console.error("The object \"" + objectName + "\" already exists.");
                return;
            }
            
            this.addName(objectName);

            let listItem = CosmoScout.gui.loadTemplateContent('simple-objects-list-item');
            listItem.innerHTML = listItem.innerHTML.replace(/%NAME%/g, objectName).replace(/%ID%/g, objectName).trim();
            listItem.id = `simple-object-id-${this.formatName(objectName)}`;

            document.getElementById('simple-objects-list').appendChild(listItem);

            CosmoScout.gui.initTooltips();

            this._sortObjectList(document.getElementById('simple-objects-list'));
        }

        /**
         * Removes the corresponding entry from the list of objects
         * @param {string} objectName   Unformatted name of the object
         * @returns                     If the object does not exist
         */
        removeObjectFromList(objectName) {
            
            if(!this.hasName(objectName)) {
                console.error("The object \"" + objectName + "\" does not exist.");
                return;
            }
            
            this.removeName(objectName)
            let object = document.querySelector("#simple-object-id-" + this.formatName(objectName));
            if (object) {
                object.remove();
            }
        }

// ------------ Helper functions for registered names ---------------------

        /**
         * Return true if the object name is registered
         * @param {string} objectName (can be unformatted)
         * @returns {Boolean}
         */
        hasName(objectName) {
            return this._objectNames.has(this.formatName(objectName));
        }

        /**
         * Adds the name to the set of known objects names
         * @param {string} objectName 
         */
        addName(objectName) {
            this._objectNames.add(this.formatName(objectName));
        }

        /**
         * Removes the name from the set of known objects names
         * @param {string} objectName 
         */
        removeName(objectName) {
            this._objectNames.delete(this.formatName(objectName));
        }

        /**
         * Formats the object name. This removes all whitespaces.
         * @param {string} objectName 
         * @returns object name without whitespaces
         */
        formatName(objectName) {
            return objectName.replace(/\s/g,'');
        }


// ------------ Input validation ----------------------------------------

        /**
         * Validates all input fields. Checks are made in following order: 
         * 
         * (*) only executed if checkFormatOnly is false
         * 
         * - are required fields given? (*)
         * - whether name is still unused
         * - location inputs -> all or none
         * - rotation inputs -> all or none 
         * - name format (\"<> symbols are not allowed)
         * - format of all numbers
         * 
         * @param {boolean} checkFormatOnly 
         * @returns {boolean} returns true if all fields are valid.
         */
        validateInput(checkFormatOnly = false) {
            
            this._resetInvalid();

            let allCorrect = true;

            let markInvalid = (div, isInvalid, message) => {
                if (isInvalid) {
                    div.classList.add("is-invalid");
                    div.parentNode.querySelector(".invalid-feedback").textContent = message;
                    allCorrect = false;
                } else {
                    div.classList.remove("is-invalid");
                }
            };

            let nameGiven = this._nameDiv.value != "";
            let anyLocationGiven = this._locationLngDiv.value != "" || this._locationLngDiv.value != "";
            let anyRotationGiven = this._rotationXDiv.value != "" || this._rotationYDiv.value != ""
                                || this._rotationZDiv.value != "" || this._rotationWDiv.value != "";

            // Check for completeness first, then for correct formatting.
            if(!checkFormatOnly) {    
                
                let anchorGiven = this._anchorDiv.value != "";
                let modelGiven = !(this._modelSelect.value == "-1" || this._modelSelect.value == "");
                let mapGiven = !(this._mapSelect.value == "-1" || this._mapSelect.value == "");
                
                let locationGiven = this._locationLngDiv.value != "" && this._locationLngDiv.value != "";         


                if (!(nameGiven || anchorGiven || locationGiven || mapGiven || modelGiven)) {

                    let markInvalid = (div) => {
                        div.classList.add("is-invalid");
                        div.parentNode.querySelector(".invalid-feedback").textContent = "";
                    }

                    markInvalid(this._nameDiv);
                    markInvalid(this._anchorDiv);
                    markInvalid(this._mapButton);
                    markInvalid(this._modelButton);
                    markInvalid(this._locationLngDiv);
                    markInvalid(this._locationLatDiv);

                    this._nothingGivenError.style.display = "block";
                    allCorrect = false;

                } else {

                    markInvalid(this._nameDiv, !nameGiven, "Please choose a name.");
                    markInvalid(this._anchorDiv, !anchorGiven, "Please choose an anchor.");
                    markInvalid(this._mapButton, !mapGiven, "Please select a map.");
                    markInvalid(this._modelButton, !modelGiven, "Please select a model.");

                    if (!anyLocationGiven) {
                        markInvalid(this._locationLatDiv, true, "");
                        markInvalid(this._locationLngDiv, true, "Set a location.");
                    }

                    this._nothingGivenError.style.display = "none";
                }
            }

            // Now CHECK FORMATTING of the given values.

            // Print error if an object with given name already exists
            if(nameGiven) {
                markInvalid(this._nameDiv, 
                    this.hasName(this._nameDiv.value) && (this._editObjectName == null || this._editObjectName != this._nameDiv.value), 
                    "This name is already in use.");
            }
            
            if(anyLocationGiven) {
                markInvalid(this._locationLngDiv, this._locationLngDiv.value == "", "Both values are required.");
                markInvalid(this._locationLatDiv, this._locationLatDiv.value == "", "Both values are required.");
            }

            if(anyRotationGiven) {
                let errorMsg = "Provide all rotation values or none.";
                markInvalid(this._rotationXDiv, this._rotationXDiv.value == "", errorMsg);
                markInvalid(this._rotationYDiv, this._rotationYDiv.value == "", errorMsg);
                markInvalid(this._rotationZDiv, this._rotationZDiv.value == "", errorMsg);
                markInvalid(this._rotationWDiv, this._rotationWDiv.value == "", errorMsg);
            }

            // Abort if errors occurred.
            if(!allCorrect) return false;

            var textRegex = /[\\"<>]/;
            let textError = "This shouldn't contain special characters like \\, \", < or >.";
            markInvalid(this._nameDiv, this._nameDiv.value.match(textRegex), textError);

            let numRegex = /^[-+]?[0-9]+(\.[0-9]*)?$/;
            let numError = "Must be a number.";

            if(anyLocationGiven) {
                markInvalid(this._locationLngDiv, !this._locationLngDiv.value.match(numRegex), numError);
                markInvalid(this._locationLatDiv, !this._locationLatDiv.value.match(numRegex), numError);
            }
            
            if(this._elevationDiv.value) {
                markInvalid(this._elevationDiv, !this._elevationDiv.value.match(numRegex), numError);
            }
            if(this._scaleDiv.value) {
                markInvalid(this._scaleDiv, !this._scaleDiv.value.match(numRegex), numError);
            }
            if(this._sizeDiv.value) {
                markInvalid(this._sizeDiv, !this._sizeDiv.value.match(numRegex), numError);
            }
            if (anyRotationGiven) {
                markInvalid(this._rotationXDiv, !this._rotationXDiv.value.match(numRegex), numError);
                markInvalid(this._rotationYDiv, !this._rotationYDiv.value.match(numRegex), numError);
                markInvalid(this._rotationZDiv, !this._rotationZDiv.value.match(numRegex), numError);
                markInvalid(this._rotationWDiv, !this._rotationWDiv.value.match(numRegex), numError);
            }

            return allCorrect;
        }


        /**
         * Generates a JSON object holding the given configuration oth the simple object.
         * @param {boolean} deleteFromList idk, why I implemented this. Maybe it can be deleted.
         * @returns {object} JSON object holding the object configuration
         */
        generateJSONObject(deleteFromList = true) {
            let object = { 
                modelFile: "../share/resources/models/" + this._modelSelect.value, 
                environmentMap: "../share/resources/textures/" + this._mapSelect.value,
                anchor: this._anchorDiv.value,
                lngLat: [
                    parseFloat(this._locationLngDiv.value),
                    parseFloat(this._locationLatDiv.value),
                ]
            };

            if(this._elevationDiv.value) {
                object.elevation = parseFloat(this._elevationDiv.value);
            }

            if(this._scaleDiv.value) {
                object.scale = parseFloat(this._scaleDiv.value);
            }

            if(this._sizeDiv.value) {
                object.diagonalLength = parseFloat(this._sizeDiv.value);
            }

            if (this._rotationXDiv.value || this._rotationYDiv.value || this._rotationZDiv.value || this._rotationWDiv.value) {
                object.rotation = [
                    parseFloat(this._rotationXDiv.value), parseFloat(this._rotationYDiv.value),
                    parseFloat(this._rotationZDiv.value), parseFloat(this._rotationWDiv.value)
                ];
            }

            if(this._alignSurfaceDiv.checked) {
                object.alignToSurface = true;
            }

            return object;
        }


// ------------ Functions for picking a location on the ground -----------

        /**
         * Toggles the availability of the pick location functionality automatically based on the height of the observer.
         * Only if the observer is closer to the ground than minHeight, the button is clickable.
         * @returns If the editor is not open
         */
        updatePickLocationButton() {
            if (!this._editorOpen) return;

            let pos = CosmoScout.state.observerLngLatHeight;
            if (pos !== undefined) {

                const minHeight = 1500;

                if(pos[2] <= minHeight) {
                    this._pickLocationButton.classList.remove("disabled");
                    this._lastHeight = pos[2];
                } else if (this._lastHeight <= minHeight) { 
                    this._pickLocationButton.classList.add("disabled");
                    this.setPickLocationEnabled(false);
                    this._lastHeight = pos[2];
                }
            }   
        }
        
        /**
         * Enables/disables the pick location functionality in the backend and the button in the editor.
         * @param {boolean} enable 
         */
        setPickLocationEnabled(enable) {

            this._pickLocationButtonActive = enable;
            CosmoScout.callbacks.simpleObjects.setPickLocationEnabled(enable);
            
            if(!enable) {
                this._pickLocationButton.classList.remove("active");
            }
        }

        /**
         * Sets the location, anchor name and disables the pick location button.
         * Is usually called from the plugin backend after the location was picked.
         * @param {number|string} lng      Longitude coordinate
         * @param {number|string} lat      Latitude coordinate
         * @param {string} anchor   Anchor name
         */
        setLngLatAnchor(lng, lat, anchor) {
            this._locationLngDiv.value = lng;
            this._locationLatDiv.value = lat;
            this._anchorDiv.value = anchor;
            this.setPickLocationEnabled(false);

            if(this.validateInput(true)) { this.updateModel(); }
        }




// ------- Private utility funcitons for the editor window -----------

        /**
         * Displays the editor window, sets the title and disables the pick location button.
         * @param {string} title 
         * @private
         */
        _openEditor(title) {
            this._title.textContent = title;
            this._editor.classList.add("visible");
            this._pickLocationButton.classList.add("disabled");
            this._editorOpen = true;
            this.setPickLocationEnabled(false);
        }

        /**
         * Hides the editor window
         * @private
         */
        _closeEditor() {
            this._editor.classList.remove("visible");
            this._editorOpen = false;
        }

        /**
         * Clears all input fields
         * @private
         */
        _resetFields() {
            this._nameDiv.value = "";
            this._anchorDiv.value = "";
            this._modelSelect.value = "-1";
            this._mapSelect.value = "-1";
            this._modelButton.firstChild.firstChild.firstChild.innerHTML = this._modelSelect.options[this._modelSelect.selectedIndex].text;
            this._mapButton.firstChild.firstChild.firstChild.innerHTML = this._mapSelect.options[this._mapSelect.selectedIndex].text;
            this._locationLngDiv.value = "";
            this._locationLatDiv.value = "";
            this._elevationDiv.value = "";
            this._scaleDiv.value = "";
            this._sizeDiv.value = "";
            this._rotationXDiv.value = "";
            this._rotationYDiv.value = "";
            this._rotationZDiv.value = "";
            this._rotationWDiv.value = "";
            this._alignSurfaceDiv.checked = false;
            CosmoScout.callbacks.simpleObjects.setAlignToSufaceEnabled(false);
        }

        /**
         * Resets all invalid markers of the input fields.
         * @private
         */
        _resetInvalid() {
            let removeFrom = (div) => {
                div.classList.remove("is-invalid");
                div.parentNode.querySelector(".invalid-feedback").textContent = "";
            }

            removeFrom(this._nameDiv);
            removeFrom(this._anchorDiv);
            removeFrom(this._modelButton);
            removeFrom(this._mapButton);
            removeFrom(this._locationLngDiv);
            removeFrom(this._locationLatDiv);
            removeFrom(this._elevationDiv);
            removeFrom(this._scaleDiv);
            removeFrom(this._sizeDiv);
            removeFrom(this._rotationXDiv);
            removeFrom(this._rotationYDiv);
            removeFrom(this._rotationZDiv);
            removeFrom(this._rotationWDiv);

            this._nothingGivenError.style.display = "none";
        }

        /**
         * Sorts the list of simple objects alphabetically.
         * @param {HTMLElement} container The element containing simpleobject divs.
         * @private
         */
        _sortObjectList(container) {
            Array.prototype.slice.call(container.children)
                .sort((ea, eb) => {
                    let a = ea.querySelector(".simpleobjects-name").textContent;
                    let b = eb.querySelector(".simpleobjects-name").textContent;
                    return a < b ? -1 : (a > b ? 1 : 0);
                })
                .forEach((div) => {
                container.appendChild(div);
            });
        }
    }

    CosmoScout.init(SimpleObjectsEditorApi);
})();