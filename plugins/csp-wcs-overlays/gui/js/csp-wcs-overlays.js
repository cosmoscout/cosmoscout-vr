/* global IApi, CosmoScout, $ */

(() => {
  /**
   * WCS overlays Api
   */
  class WCSOverlaysApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'wcsOverlays';

    /**
     * @inheritDoc
     */
    init() {
      this._coverageSelect = document.querySelector(`[data-callback="wcsOverlays.setCoverage"]`);

      this._infoWindow = CosmoScout.gui.loadTemplateContent("wcsOverlays-infoWindow");
      document.getElementById("cosmoscout").appendChild(this._infoWindow);
      this._defaultBoundsLabel = document.getElementById("wcsOverlays-defaultBounds");
      this._defaultBoundsGoTo  = document.querySelector(
          '[onclick="CosmoScout.callbacks.wcsOverlays.goToDefaultBounds()"]');
      this._currentBoundsLabel = document.getElementById("wcsOverlays-currentBounds");
      this._currentBoundsGoTo  = document.querySelector(
          '[onclick="CosmoScout.callbacks.wcsOverlays.goToCurrentBounds()"]');
      this._currentBoundsUpdate =
          document.querySelector('[onclick="CosmoScout.callbacks.wcsOverlays.updateBounds()"]');
      this._currentBoundsReset =
          document.querySelector('[onclick="CosmoScout.callbacks.wcsOverlays.resetBounds()"]');

      this._firstTime =
          document.querySelector('[onclick="CosmoScout.callbacks.wcsOverlays.goToFirstTime()"]');
      this._previousTime =
          document.querySelector('[onclick="CosmoScout.callbacks.wcsOverlays.goToPreviousTime()"]');
      this._nextTime =
          document.querySelector('[onclick="CosmoScout.callbacks.wcsOverlays.goToNextTime()"]');
      this._lastTime =
          document.querySelector('[onclick="CosmoScout.callbacks.wcsOverlays.goToLastTime()"]');

      this._infoIcon        = document.getElementById("wcsOverlays-infoIcon");
      this._infoTitle       = document.getElementById("wcsOverlays-infoWindow-title");
      this._infoAbstract    = document.getElementById("wcsOverlays-infoWindow-abstract");
      this._infoAttribution = document.getElementById("wcsOverlays-infoWindow-attribution");
      this._infoKeywords    = document.getElementById("wcsOverlays-infoWindow-keywords");

      this._transferFunction = CosmoScout.transferFunctionEditor.create(
          document.getElementById('wcsOverlays-transfer-function'), transferFunction => {
            window.callNative('wcsOverlays.setTransferFunction', transferFunction);
          }, {width: 450, height: 120, defaultFunction: "HeatLight.json", fitToData: true});

      CosmoScout.gui.initSlider("wcsOverlays.setUpdateBoundsDelay", 0, 5000, 100, [1000]);
      CosmoScout.gui.initSlider("wcsOverlays.setPrefetchCount", 0, 10, 1, [0]);
      CosmoScout.gui.initSliderOptions("wcsOverlays.setMaxTextureSize", {
        start: [1024],
        connect: "lower",
        range: {'min': 256, '25%': 512, '50%': 1024, '75%': 2048, 'max': 4096},
        snap: true,
        format: {
          to(value) {
            return CosmoScout.utils.beautifyNumber(value);
          },
          from(value) {
            return Number(parseFloat(value));
          },
        },
      });
    }

    /**
     * Set layer info that is shown in the info window.
     *
     * @param {string} title The human readable title of the layer
     * @param {string} abstract A narrative description of the layer
     * @param {string} attribution Attribution of the shown data
     */
    setInfo(title, abstract, attribution, keywords) {
      this._infoTitle.innerHTML       = title;
      this._infoAbstract.innerHTML    = abstract;
      this._infoAttribution.innerHTML = attribution;
      this._infoKeywords.innerHTML    = keywords;
    }

    /**
     * Enable or disable the button for showing the info window.
     * If the button is disabled, the window will also be hidden.
     *
     * @param {boolean} enable The desired state of the button
     */
    enableInfoButton(enable) {
      if (!enable) {
        this._infoWindow.classList.remove('visible');
      }

      this._infoIcon.onclick =
          (enable ? () => {this._infoWindow.classList.toggle('visible')} : () => {});
    }

    /**
     * Recreate the dropdown for layer selection
     *
     * When the dropdown contains a lot of values, bootstrap-select apparently adds margins to the
     * dropdown menu, that are not removed when switching to a server with fewer layers. To remove
     * the margins the selectpicker can be recreated using this function.
     */
    resetLayerSelect() {
      $(this._coverageSelect).selectpicker("destroy");
      $(this._coverageSelect).selectpicker();
      CosmoScout.gui.clearDropdown("wcsOverlays.setCoverage");
      CosmoScout.gui.addDropdownValue("wcsOverlays.setCoverage", "None", "None", false);

      $(document.querySelector(`[data-callback="wcsOverlays.setLayer"]`)).selectpicker("destroy");
      $(document.querySelector(`[data-callback="wcsOverlays.setLayer"]`)).selectpicker();
    }

    /**
     * Call the refresh function on the layer dropdown.
     */
    refreshLayerSelect() {
      $(this._coverageSelect).selectpicker("refresh");
    }

    /**
     * Add a new layer to the layer dropdown.
     * Indent it according to the layer hierarchy.
     *
     * @param {string} name The internal name of the layer
     * @param {string} title The human readable title of the layer
     * @param {boolean} active Whether the layer is currently selected
     * @param {boolean} requestable Whether maps for this layer may be requested
     * @param {number} depth The depth of the layer in the hierarchy
     */
    addCoverage(name, title, active, requestable, depth) {
      const option = document.createElement('option');

      option.value     = name;
      option.selected  = active;
      option.disabled  = !requestable;
      option.innerHTML = "&emsp;".repeat(depth) + title;

      this._coverageSelect.appendChild(option);
    }

    /**
     * Set the default bounds for the active layer.
     * Also enable the "Go to center" button.
     *
     * @param {number|string} minLon
     * @param {number|string} maxLon
     * @param {number|string} minLat
     * @param {number|string} maxLat
     */
    setDefaultBounds(minLon, maxLon, minLat, maxLat) {
      this._defaultBoundsLabel.innerText = `${CosmoScout.utils.formatLongitude(minLon)}, ` +
                                           `${CosmoScout.utils.formatLatitude(minLat)} - ` +
                                           `${CosmoScout.utils.formatLongitude(maxLon)}, ` +
                                           `${CosmoScout.utils.formatLatitude(maxLat)}`;
      this._defaultBoundsGoTo.disabled = false;
    }

    /**
     * Set the current bounds for the active layer.
     * Also enable the "Go to center" and "Update bounds" buttons.
     *
     * @param {number|string} minLon
     * @param {number|string} maxLon
     * @param {number|string} minLat
     * @param {number|string} maxLat
     */
    setCurrentBounds(minLon, maxLon, minLat, maxLat) {
      this._currentBoundsLabel.innerText = `${CosmoScout.utils.formatLongitude(minLon)}, ` +
                                           `${CosmoScout.utils.formatLatitude(minLat)} - ` +
                                           `${CosmoScout.utils.formatLongitude(maxLon)}, ` +
                                           `${CosmoScout.utils.formatLatitude(maxLat)}`;
      this._currentBoundsGoTo.disabled = false;
      this._currentBoundsUpdate.disabled = false;
      this._currentBoundsReset.disabled  = false;
    }

    /**
     * Display no default bounds.
     * Also disable the "Go to center" button.
     */
    clearDefaultBounds() {
      this._defaultBoundsLabel.innerText = "None";
      this._defaultBoundsGoTo.disabled   = true;
    }

    /**
     * Display no current bounds.
     * Also disable the "Go to center" and "Update bounds" buttons.
     */
    clearCurrentBounds() {
      this._currentBoundsLabel.innerText = "None";
      this._currentBoundsGoTo.disabled   = true;
      this._currentBoundsUpdate.disabled = true;
      this._currentBoundsReset.disabled  = true;
    }

    /**
     * Enable or disable the time controls.
     *
     * @param {boolean} enable The desired state for the time controls
     */
    enableTimeNavigation(enable) {
      this._firstTime.disabled    = !enable;
      this._previousTime.disabled = !enable;
      this._nextTime.disabled     = !enable;
      this._lastTime.disabled     = !enable;
    }

    /**
     * Set the min and max range of the texture
     * @param {Number} min
     * @param {Number} max
     */
    setDataRange(min, max) {
      if (typeof this._transferFunction !== 'object') {
        return;
      }

      this._range = [min, max];
      this.resetTransferFunction();

      this._transferFunction.setData(this._range);
    }

    /**
     * Resets the transfer function to default heat light
     */
    resetTransferFunction() {
      if (typeof this._transferFunction !== 'object') {
        return;
      }

      CosmoScout.callbacks.transferFunctionEditor.importTransferFunction(
          "HeatLight.json", this._transferFunction.id);
    }

    /**
     * Parses passed time string to be displayed in the sidebar and info window
     * @param {String} start
     * @param {String} end
     */
    setTimeInfo(start, end) {
      const container     = document.querySelector('#wcsOverlays-time-info-container');
      const containerInfo = document.querySelector('#wcsOverlays-infoWindow-time-container');
      const startEle      = document.querySelector('#wcsOverlays-start-time');
      const endEle        = document.querySelector('#wcsOverlays-end-time');

      const dateToString =
          dateString => {
            // The .000000Z part seems to confuse the ES Parser
            const parsed = new Date(Date.parse(dateString.split('.').shift()));
            const pad = input => (String(input)).length < 2 ? `0${String(input)}` : String(input);

            return `${parsed.getFullYear()}-${pad(parsed.getMonth() + 1)}-${
                pad(parsed.getDate())} ${pad(parsed.getHours())}:${pad(parsed.getMinutes())}:${
                pad(parsed.getSeconds())}`
          }

      if (start.length > 0) {
        startEle.textContent = dateToString(start);
      }

      if (end.length > 0) {
        endEle.textContent = dateToString(end);
      }

      if (start.length + end.length > 0) {
        container.classList.remove('hidden');
        containerInfo.classList.remove('hidden');
        document.querySelector('#wcsOverlays-infoWindow-time').innerHTML =
            `${startEle.textContent}&ndash;${endEle.textContent}`;
      } else {
        container.classList.add('hidden');
        containerInfo.classList.add('hidden');
      }
    }

    /**
     * Enables or disables the select picker
     * Automatically called when a texture is currently loading
     *
     * @param {boolean} state
     */
    setCoverageSelectDisabled(state) {
      // Intentional
      if (state == true) {
        this._coverageSelect.setAttribute('disabled', '1');
      } else {
        this._coverageSelect.removeAttribute('disabled');
      }
      $(this._coverageSelect).selectpicker('refresh');
    }

    setNumberOfLayers(layers, active) {
      CosmoScout.gui.clearDropdown('wcsOverlays.setLayer');

      for (let i = 1; i < layers + 1; i++) {
        CosmoScout.gui.addDropdownValue(
            'wcsOverlays.setLayer', i, `Layer ${i}`, i === Number(active));
      }
    }
  }

  CosmoScout.init(WCSOverlaysApi);
})();
