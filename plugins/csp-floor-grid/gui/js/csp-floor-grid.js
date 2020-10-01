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
            CosmoScout.gui.initSlider("floorGrid.setSize", -5, 5, 1, [0]);
            CosmoScout.gui.initSlider("floorGrid.setOffset", -3, 0, 0.01, [-1.8]);
            CosmoScout.gui.initSlider("floorGrid.setAlpha", 0, 1, 0.01, [1]);
            //CosmoScout.gui.initColorPickers()
        }
    }

    CosmoScout.init(FloorGridApi);
})();
