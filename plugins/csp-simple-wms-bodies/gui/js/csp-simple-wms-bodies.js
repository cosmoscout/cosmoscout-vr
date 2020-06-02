/* global IApi, CosmoScout, $ */

(() => {
  /**
   * Simple WMS bodies Api
   */
  class SimpleWMSBodiesApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'simpleWMSBodies';

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
      document.querySelector('[data-callback="simpleWMSBodies.setEnableTimeSpan"]').disabled =
          !enable;
    }
  }

  CosmoScout.init(SimpleWMSBodiesApi);
})();
