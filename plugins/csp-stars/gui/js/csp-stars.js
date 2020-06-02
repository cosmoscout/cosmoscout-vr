/* global IApi, CosmoScout */

(() => {
  /**
   * Stars Api
   */
  class StarsApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'stars';

    /**
     * @inheritDoc
     */
    init() {
      CosmoScout.gui.initSlider("stars.setMagnitude", -10.0, 20.0, 0.1, [-5, 15]);
      CosmoScout.gui.initSlider("stars.setSize", 0.01, 1, 0.01, [0.05]);
      CosmoScout.gui.initSlider("stars.setLuminanceBoost", 0.0, 20.0, 0.1, [0]);
    }
  }

  CosmoScout.init(StarsApi);
})();