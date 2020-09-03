/* global IApi, CosmoScout */

(() => {
  /**
   * Atmosphere Api
   */
  class AtmosphereApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'atmosphere';

    /**
     * @inheritDoc
     */
    init() {
      CosmoScout.gui.initSlider("atmosphere.setQuality", 1, 30, 1, [7]);
      CosmoScout.gui.initSlider("atmosphere.setWaterLevel", -2, 2, 0.01, [0]);
    }
  }

  CosmoScout.init(AtmosphereApi);
})();
