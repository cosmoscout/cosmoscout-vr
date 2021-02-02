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
      this._infoWindow = CosmoScout.gui.loadTemplateContent("wms-info");
      document.getElementById("cosmoscout").appendChild(this._infoWindow);

      this._layerSelect = document.querySelector(`[data-callback="wmsOverlays.setLayer"]`);

      this._defaultBoundsLabel = document.getElementById("wmsOverlays.defaultBounds");
      this._defaultBoundsGoTo  = document.querySelector(
          '[onclick="CosmoScout.callbacks.wmsOverlays.goToDefaultBounds()"]');
      this._currentBoundsLabel = document.getElementById("wmsOverlays.currentBounds");
      this._currentBoundsGoTo  = document.querySelector(
          '[onclick="CosmoScout.callbacks.wmsOverlays.goToCurrentBounds()"]');
      this._currentBoundsUpdate =
          document.querySelector('[onclick="CosmoScout.callbacks.wmsOverlays.updateBounds()"]');

      this._firstTime =
          document.querySelector('[onclick="CosmoScout.callbacks.wmsOverlays.goToFirstTime()"]');
      this._previousTime =
          document.querySelector('[onclick="CosmoScout.callbacks.wmsOverlays.goToPreviousTime()"]');
      this._nextTime =
          document.querySelector('[onclick="CosmoScout.callbacks.wmsOverlays.goToNextTime()"]');
      this._lastTime =
          document.querySelector('[onclick="CosmoScout.callbacks.wmsOverlays.goToLastTime()"]');

      this._infoIcon        = document.getElementById("wmsOverlays.infoIcon");
      this._infoTitle       = document.getElementById("wmsOverlays.infoWindow.title");
      this._infoAbstract    = document.getElementById("wmsOverlays.infoWindow.abstract");
      this._infoAttribution = document.getElementById("wmsOverlays.infoWindow.attribution");

      this._scaleLabel   = document.getElementById("wmsOverlays.scale");
      this._scaleWarning = document.getElementById("wmsOverlays.scaleWarning");

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

    setScale(value) {
      this._scaleLabel.innerHTML = "1:" + CosmoScout.utils.formatNumber(value);
    }

    showScaleWarning(enable, text = "") {
      this._scaleWarning.style.visibility = (enable ? "visible" : "hidden");
      $(this._scaleWarning).tooltip({placement: "top"}).attr("data-original-title", text);
    }

    setInfo(title, abstract, attribution) {
      this._infoTitle.innerHTML       = title;
      this._infoAbstract.innerHTML    = abstract;
      this._infoAttribution.innerHTML = attribution;
    }

    enableInfoButton(enable) {
      if (!enable) {
        this._infoWindow.classList.remove('visible');
      }
      this._infoIcon.onclick =
          (enable ? () => { CosmoScout.callbacks.wmsOverlays.showInfo(); } : () => { return; });
    }

    /**
     * Recreates the dropdown for layer selection
     *
     * When the dropdown contains a lot of values, bootstrap-select apparently adds margins to the
     * dropdown menu, that are not removed when switching to a server with fewer layers. To remove
     * the margins the selectpicker can be recreated using this function.
     */
    resetLayerSelect() {
      $(this._layerSelect).selectpicker("destroy");
      $(this._layerSelect).selectpicker();
    }

    refreshLayerSelect() {
      $(this._layerSelect).selectpicker("refresh");
    }

    addLayer(name, title, active, requestable, depth) {
      const option = document.createElement('option');

      option.value     = name;
      option.selected  = active;
      option.disabled  = !requestable;
      option.innerHTML = "&emsp;".repeat(depth) + title;

      this._layerSelect.appendChild(option);
    }

    setLegendURL(url) {
      document.getElementById("wmsOverlays.legend").src = url;
    }

    setDefaultBounds(minLon, maxLon, minLat, maxLat) {
      this._defaultBoundsLabel.innerText = `${CosmoScout.utils.formatLongitude(minLon)}, ` +
                                           `${CosmoScout.utils.formatLatitude(minLat)} - ` +
                                           `${CosmoScout.utils.formatLongitude(maxLon)}, ` +
                                           `${CosmoScout.utils.formatLatitude(maxLat)}`;
      this._defaultBoundsGoTo.disabled = false;
    }

    setCurrentBounds(minLon, maxLon, minLat, maxLat) {
      this._currentBoundsLabel.innerText = `${CosmoScout.utils.formatLongitude(minLon)}, ` +
                                           `${CosmoScout.utils.formatLatitude(minLat)} - ` +
                                           `${CosmoScout.utils.formatLongitude(maxLon)}, ` +
                                           `${CosmoScout.utils.formatLatitude(maxLat)}`;
      this._currentBoundsGoTo.disabled = false;
      this._currentBoundsUpdate.disabled = false;
    }

    clearDefaultBounds() {
      this._defaultBoundsLabel.innerText = "None";
      this._defaultBoundsGoTo.disabled   = true;
    }

    clearCurrentBounds() {
      this._currentBoundsLabel.innerText = "None";
      this._currentBoundsGoTo.disabled   = true;
      this._currentBoundsUpdate.disabled = true;
    }

    setNoSubsets() {
      this._currentBoundsLabel.innerText = "No subsets allowed for this layer";
      this._currentBoundsGoTo.disabled   = true;
      this._currentBoundsUpdate.disabled = true;
    }

    enableTimeNavigation(enable) {
      this._firstTime.disabled    = !enable;
      this._previousTime.disabled = !enable;
      this._nextTime.disabled     = !enable;
      this._lastTime.disabled     = !enable;
    }
  }

  CosmoScout.init(WMSOverlaysApi);
})();
