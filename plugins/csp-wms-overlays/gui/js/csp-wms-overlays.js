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
      document.querySelector('[data-callback="wmsOverlays.setEnableTimeSpan"]').disabled =
          !enable;
    }
  }

  CosmoScout.init(WMSOverlaysApi);
})();
