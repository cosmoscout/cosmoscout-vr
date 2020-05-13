/* global IApi, CosmoScout, $ */

(() => {
  /**
   * LOD Bodies Api
   */
  class LODBodiesApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'lodBodies';

    /**
     * @inheritDoc
     */
    init() {
      CosmoScout.gui.initSlider('lodBodies.setHeightRange', -12.0, 21.0, 0.1, [-8, 12]);
      CosmoScout.gui.initSlider('lodBodies.setSlopeRange', 0.0, 90.0, 1.0, [0, 45]);
      CosmoScout.gui.initSlider('lodBodies.setTerrainLod', 10.0, 50.0, 0.1, [15]);
      CosmoScout.gui.initSlider('lodBodies.setTextureGamma', 0.1, 3.0, 0.01, [1.0]);

      const terrainLod = document.querySelector('[data-callback="lodBodies.setTerrainLod"]');
      document.querySelector('[data-callback="lodBodies.setEnableAutoTerrainLod"]')
          .addEventListener('change', (event) => {
            if (event.target.checked) {
              terrainLod.classList.add('unresponsive');
            } else {
              terrainLod.classList.remove('unresponsive');
            }
          });
    }

    /**
     * Sets an elevation data copyright tooltip
     * TODO Remove jQuery
     *
     * @param copyright {string}
     */
    // eslint-disable-next-line class-methods-use-this
    setElevationDataCopyright(copyright) {
      $('#lodbodies-dem-data-copyright')
          .tooltip({placement: 'top'})
          .attr('data-original-title', `© ${copyright}`);
    }

    /**
     * Sets a map data copyright tooltip
     * TODO Remove jQuery
     *
     * @param copyright {string}
     */
    // eslint-disable-next-line class-methods-use-this
    setMapDataCopyright(copyright) {
      $('#lodbodies-img-data-copyright')
          .tooltip({placement: 'bottom'})
          .attr('data-original-title', `© ${copyright}`);
    }
  }

  CosmoScout.init(LODBodiesApi);
})();
