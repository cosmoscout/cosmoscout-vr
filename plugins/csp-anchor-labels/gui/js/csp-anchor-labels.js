/* global IApi, CosmoScout */

(() => {
  /**
   * Atmosphere Api
   */
  class AnchorLabelApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'anchorLabels';

    /**
     * @inheritDoc
     */
    init() {
      CosmoScout.gui.initSlider('anchorLabels.setIgnoreOverlapThreshold', 0.0, 0.2, 0.001, [0.1]);
      CosmoScout.gui.initSlider('anchorLabels.setScale', 0.1, 5.0, 0.01, [1.2]);
      CosmoScout.gui.initSlider('anchorLabels.setDepthScale', 0.0, 1.0, 0.01, [1.0]);
      CosmoScout.gui.initSlider('anchorLabels.setOffset', 0.0, 1.0, 0.01, [0.2]);
    }
  }

  CosmoScout.init(AnchorLabelApi);
})();
