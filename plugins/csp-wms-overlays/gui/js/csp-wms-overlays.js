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
     * Enables or disables timespan checkbox
     *
     * @param enable {bool}
     */
    // eslint-disable-next-line class-methods-use-this
    enableCheckBox(enable) {
      document.querySelector('[data-callback="wmsOverlays.setEnableTimeSpan"]').disabled = !enable;
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
    }
  }

  CosmoScout.init(WMSOverlaysApi);
})();