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
            const pickerDiv = document.querySelector('#floorGrid-setColor');

            pickerDiv.picker = new CP(pickerDiv);
            pickerDiv.picker.self.classList.add('no-alpha');
            pickerDiv.picker.on('change', (r, g, b, a) => {
                const color                = CP.HEX([r, g, b, 1]);
                pickerDiv.style.background = color;
                pickerDiv.value            = color;
                CosmoScout.callbacks.floorGrid.setColor(color);
            });

            pickerDiv.oninput = (e) => {
                const color = CP.HEX(e.target.value);
                pickerDiv.picker.set(color[0], color[1], color[2], 1);
                pickerDiv.style.background = CP.HEX([color[0], color[1], color[2], 1]);
                CosmoScout.callbacks.floorGrid.setColor(color);
            };
        }
    }

    CosmoScout.init(FloorGridApi);
})();
