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
    }

    /**
     * Sets an elevation data copyright tooltip
     * TODO Remove jQuery
     *
     * @param copyright {string}
     */
    // eslint-disable-next-line class-methods-use-this
    setWMSDataCopyright(copyright) {
      $('#wms-img-data-copyright')
          .tooltip({placement: 'top'})
          .attr('data-original-title', `© ${copyright}`);
    }

    /**
     * Recreates the dropdown for layer selection
     *
     * When the dropdown contains a lot of values, bootstrap-select apparently adds margins to the
     * dropdown menu, that are not removed when switching to a server with fewer layers. To remove
     * the margins the selectpicker can be recreated using this function.
     */
    resetLayerSelect() {
      const dropdown = document.querySelector(`[data-callback="wmsOverlays.setLayer"]`);
      $(dropdown).selectpicker("destroy");
      $(dropdown).selectpicker();
    }

    refreshLayerSelect() {
      const dropdown = document.querySelector(`[data-callback="wmsOverlays.setLayer"]`);
      $(dropdown).selectpicker("refresh");
    }

    addLayer(name, title, active, requestable, depth) {
      const dropdown = document.querySelector(`[data-callback="wmsOverlays.setLayer"]`);
      const option   = document.createElement('option');

      option.value     = name;
      option.selected  = active;
      option.disabled  = !requestable;
      option.innerHTML = "&emsp;".repeat(depth) + title;

      dropdown.appendChild(option);
    }

    setLegendURL(url) {
      document.getElementById("wmsOverlays.legend").src = url;
    }

    setDefaultBounds(minLon, maxLon, minLat, maxLat) {
      document.getElementById("wmsOverlays.defaultBounds").innerText =
          `${CosmoScout.utils.formatLongitude(minLon)}, ` +
          `${CosmoScout.utils.formatLatitude(minLat)} - ` +
          `${CosmoScout.utils.formatLongitude(maxLon)}, ` +
          `${CosmoScout.utils.formatLatitude(maxLat)}`;
      document.querySelector(`[onclick="CosmoScout.callbacks.wmsOverlays.goToDefaultBounds()"]`)
          .disabled = false;
    }

    setCurrentBounds(minLon, maxLon, minLat, maxLat) {
      document.getElementById("wmsOverlays.currentBounds").innerText =
          `${CosmoScout.utils.formatLongitude(minLon)}, ` +
          `${CosmoScout.utils.formatLatitude(minLat)} - ` +
          `${CosmoScout.utils.formatLongitude(maxLon)}, ` +
          `${CosmoScout.utils.formatLatitude(maxLat)}`;
      document.querySelector(`[onclick="CosmoScout.callbacks.wmsOverlays.goToCurrentBounds()"]`)
          .disabled = false;
      document.querySelector('[onclick="CosmoScout.callbacks.wmsOverlays.updateBounds()"]')
          .disabled = false;
    }

    clearDefaultBounds() {
      document.getElementById("wmsOverlays.defaultBounds").innerText = "None";
      document.querySelector(`[onclick="CosmoScout.callbacks.wmsOverlays.goToDefaultBounds()"]`)
          .disabled = true;
    }

    clearCurrentBounds() {
      document.getElementById("wmsOverlays.currentBounds").innerText = "None";
      document.querySelector(`[onclick="CosmoScout.callbacks.wmsOverlays.goToCurrentBounds()"]`)
          .disabled = true;
      document.querySelector('[onclick="CosmoScout.callbacks.wmsOverlays.updateBounds()"]')
          .disabled = true;
    }

    setNoSubsets() {
      document.getElementById("wmsOverlays.currentBounds").innerText =
          "No subsets allowed for this layer";
      document.querySelector(`[onclick="CosmoScout.callbacks.wmsOverlays.goToCurrentBounds()"]`)
          .disabled = true;
      document.querySelector('[onclick="CosmoScout.callbacks.wmsOverlays.updateBounds()"]')
          .disabled = true;
    }

    enableTimeNavigation(enable) {
      document.querySelector('[onclick="CosmoScout.callbacks.wmsOverlays.goToFirstTime()"]')
          .disabled = !enable;
      document.querySelector('[onclick="CosmoScout.callbacks.wmsOverlays.goToPreviousTime()"]')
          .disabled = !enable;
      document.querySelector('[onclick="CosmoScout.callbacks.wmsOverlays.goToNextTime()"]')
          .disabled = !enable;
      document.querySelector('[onclick="CosmoScout.callbacks.wmsOverlays.goToLastTime()"]')
          .disabled = !enable;
    }
  }

  CosmoScout.init(WMSOverlaysApi);
})();
