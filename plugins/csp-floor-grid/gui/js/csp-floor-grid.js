/* global IApi, CosmoScout */

(() => {
    /**
     * Floor Grid Api
     */
    class FloorGridApi extends IApi {
        /**
         * @inheritDoc
         */
        name = 'floorGrid';

        /**
         * @inheritDoc
         */
        init() {
            CosmoScout.gui.initSlider("floorGrid.setSize", 0.5, 2, 0.1, [1]);
            CosmoScout.gui.initSlider("floorGrid.setOffset", -3, 0, 0.01, [-1.8]);
        }
    }

    CosmoScout.init(FloorGridApi);
})();
