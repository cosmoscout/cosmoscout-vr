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
    picker;

    /**
     * @inheritDoc
     */
    init() {
      CosmoScout.gui.initSlider("floorGrid.setSize", -5, 5, 1, [0]);
      CosmoScout.gui.initSlider("floorGrid.setOffset", -3, 0, 0.01, [-1.8]);
      CosmoScout.gui.initSlider("floorGrid.setAlpha", 0, 1, 0.01, [1]);
      this.picker = document.querySelector('#floorGrid-setColor');

      this.picker.picker = new CP(this.picker);
      this.picker.picker.self.classList.add('no-alpha');
      this.picker.picker.on('drag', (r, g, b, a) => {
        const color                  = CP.HEX([r, g, b, 1]);
        this.picker.style.background = color;
        this.picker.value            = color;
        
        CosmoScout.callbacks.floorGrid.setColor(color);
      });
      this.picker.oninput = (e) => {
        const color = CP.HEX(e.target.value);
        this.picker.picker.set(color[0], color[1], color[2], 1);
        this.picker.style.background = CP.HEX([color[0], color[1], color[2], 1]);

        CosmoScout.callbacks.floorGrid.setColor(e.target.value);
      };
    }

    setColorValue(color) {
      this.picker.picker.set(CP.HEX(color));
      this.picker.style.background = color;
      this.picker.value            = color;
    }
  }

  /**
   * FoV Vignette Api
   */
  class FovVignetteApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'fovVignette';
    picker;

    /**
     * @inheritDoc
     */
    init() {
      CosmoScout.gui.initSlider("fovVignette.setInnerRadius", 0, 1, 0.01, [0.5]);
      CosmoScout.gui.initSlider("fovVignette.setOuterRadius", 0, 1.5, 0.01, [1.0]);
      CosmoScout.gui.initSlider("fovVignette.setLowerThreshold", 0, 10, 0.1, [0.2]);
      CosmoScout.gui.initSlider("fovVignette.setUpperThreshold", 0, 10, 0.1, [10]);
      CosmoScout.gui.initSlider("fovVignette.setDuration", 0, 2, 0.2, [1.0]);
      CosmoScout.gui.initSlider("fovVignette.setDeadzone", 0, 1, 0.1, [0.5]);
      this.picker = document.querySelector('#fovVignette-setColor');

      this.picker.picker = new CP(this.picker);
      this.picker.picker.self.classList.add('no-alpha');
      this.picker.picker.on('change', (r, g, b, a) => {
        const color                  = CP.HEX([r, g, b, 1]);
        this.picker.style.background = color;
        this.picker.value            = color;
        CosmoScout.callbacks.fovVignette.setColor(color);
      });

      this.picker.oninput = (e) => {
        const color = CP.HEX(e.target.value);
        this.picker.picker.set(color[0], color[1], color[2], 1);
        this.picker.style.background = CP.HEX([color[0], color[1], color[2], 1]);
        CosmoScout.callbacks.fovVignette.setColor(color);
      }
    }

    setColorValue(color) {
      this.picker.picker.set(CP.HEX(color));
      this.picker.style.background = color;
      this.picker.value            = color;
    }
  }

  CosmoScout.init(FloorGridApi, FovVignetteApi);
})();
