////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

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
      CosmoScout.gui.initSlider("atmosphere.setCloudAltitude", 0, 10000, 10, [2000]);
      CosmoScout.gui.initSlider("atmosphere.setWaterLevel", -10000, 10000, 10, [0]);
    }
  }

  CosmoScout.init(AtmosphereApi);
})();
