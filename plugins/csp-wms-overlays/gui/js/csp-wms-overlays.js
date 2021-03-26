/* global IApi, CosmoScout, $ */

(() => {
  /**
   * WMS overlays Api
   */
  class WMSOverlaysApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'wmsOverlays';

    /**
     * @inheritDoc
     */
    init() {
      this._infoWindow = CosmoScout.gui.loadTemplateContent("wmsOverlays-infoWindow");
      document.getElementById("cosmoscout").appendChild(this._infoWindow);

      this._layerSelect = document.querySelector(`[data-callback="wmsOverlays.setLayer"]`);

      this._defaultBoundsLabel = document.getElementById("wmsOverlays-defaultBounds");
      this._defaultBoundsGoTo  = document.querySelector(
          '[onclick="CosmoScout.callbacks.wmsOverlays.goToDefaultBounds()"]');
      this._currentBoundsLabel = document.getElementById("wmsOverlays-currentBounds");
      this._currentBoundsGoTo  = document.querySelector(
          '[onclick="CosmoScout.callbacks.wmsOverlays.goToCurrentBounds()"]');
      this._currentBoundsUpdate =
          document.querySelector('[onclick="CosmoScout.callbacks.wmsOverlays.updateBounds()"]');
      this._currentBoundsReset =
          document.querySelector('[onclick="CosmoScout.callbacks.wmsOverlays.resetBounds()"]');

      this._firstTime =
          document.querySelector('[onclick="CosmoScout.callbacks.wmsOverlays.goToFirstTime()"]');
      this._previousTime =
          document.querySelector('[onclick="CosmoScout.callbacks.wmsOverlays.goToPreviousTime()"]');
      this._nextTime =
          document.querySelector('[onclick="CosmoScout.callbacks.wmsOverlays.goToNextTime()"]');
      this._lastTime =
          document.querySelector('[onclick="CosmoScout.callbacks.wmsOverlays.goToLastTime()"]');

      this._legend = document.getElementById("wmsOverlays-legend");

      this._infoIcon        = document.getElementById("wmsOverlays-infoIcon");
      this._infoTitle       = document.getElementById("wmsOverlays-infoWindow-title");
      this._infoAbstract    = document.getElementById("wmsOverlays-infoWindow-abstract");
      this._infoAttribution = document.getElementById("wmsOverlays-infoWindow-attribution");

      this._scaleLabel   = document.getElementById("wmsOverlays-scale");
      this._scaleWarning = document.getElementById("wmsOverlays-scaleWarning");

      this.showScaleWarning(false);

      CosmoScout.gui.initSlider("wmsOverlays.setUpdateBoundsDelay", 0, 5000, 100, [1000]);
      CosmoScout.gui.initSlider("wmsOverlays.setPrefetchCount", 0, 10, 1, [0]);
      CosmoScout.gui.initSliderOptions("wmsOverlays.setMaxTextureSize", {
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
     * Set the displayed map scale.
     *
     * @param {number} value The map scale
     */
    setScale(value) {
      this._scaleLabel.innerHTML = "1:" + CosmoScout.utils.formatNumber(value);
    }

    /**
     * Enable or disable a warning icon for scales that aren't appropriate for the current layer.
     *
     * @param {boolean} enable Whether to enable the warning icon
     * @param {string} text Text to display in the warning icon's tooltip
     */
    showScaleWarning(enable, text = "") {
      this._scaleWarning.style.visibility = (enable ? "visible" : "hidden");
      $(this._scaleWarning).tooltip({placement: "top"}).attr("data-original-title", text);
    }

    /**
     * Set layer info that is shown in the info window.
     *
     * @param {string} title The human readable title of the layer
     * @param {string} abstract A narrative description of the layer
     * @param {string} attribution Attribution of the shown data
     */
    setInfo(title, abstract, attribution) {
      this._infoTitle.innerHTML       = title;
      this._infoAbstract.innerHTML    = abstract;
      this._infoAttribution.innerHTML = attribution;
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
          (enable ? () => {this._infoWindow.classList.toggle('visible')} : () => {
            return;
          });
    }

    /**
     * Recreate the dropdown for layer selection
     *
     * When the dropdown contains a lot of values, bootstrap-select apparently adds margins to the
     * dropdown menu, that are not removed when switching to a server with fewer layers. To remove
     * the margins the selectpicker can be recreated using this function.
     */
    resetLayerSelect() {
      $(this._layerSelect).selectpicker("destroy");
      $(this._layerSelect).selectpicker();
      CosmoScout.gui.clearDropdown("wmsOverlays.setLayer");
      CosmoScout.gui.addDropdownValue("wmsOverlays.setLayer", "None", "None", false);
    }

    /**
     * Call the refresh function on the layer dropdown.
     */
    refreshLayerSelect() {
      $(this._layerSelect).selectpicker("refresh");
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
    addLayer(name, title, active, requestable, depth) {
      const option = document.createElement('option');

      option.value     = name;
      option.selected  = active;
      option.disabled  = !requestable;
      option.innerHTML = "&emsp;".repeat(depth) + title;

      this._layerSelect.appendChild(option);
    }

    /**
     * Set the URL of the active style's legend.
     * If it is a valid URL the legend will be automatically loaded.
     *
     * @param {string} url The style's legend
     */
    setLegendURL(url) {
      this._legend.src = url;
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
      this._scaleLabel.innerHTML         = "None";
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
     * Display, that no subsets may be requested for the current layer.
     * Also disable the "Go to center" and "Update bounds" buttons.
     */
    setNoSubsets() {
      this._currentBoundsLabel.innerText = "No subsets allowed for this layer";
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
  }

  CosmoScout.init(WMSOverlaysApi);
})();
