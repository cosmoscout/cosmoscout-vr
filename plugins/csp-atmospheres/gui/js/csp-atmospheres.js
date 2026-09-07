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
      CosmoScout.gui.initSlider("atmosphere.setCloudAltitude", 0, 10000, 100, [2000]);
      CosmoScout.gui.initSlider("atmosphere.setWaterLevel", -5000, 5000, 20, [0]);
      CosmoScout.gui.initSlider("atmosphere.setCloudQuality", 0.1, 3, .1, [1]);
      CosmoScout.gui.initSlider("atmosphere.setCloudMaxSamples", 10, 20000, 10, [400]);
      CosmoScout.gui.initSlider("atmosphere.setCloudJitter", 0, 1, .01, [.5]);
      CosmoScout.gui.initSlider("atmosphere.setCloudTypeExponent", 0.01, 5, .1, [1]);
      CosmoScout.gui.initSlider("atmosphere.setCloudRangeMin", 0, 1, 0.01, [0]);
      CosmoScout.gui.initSlider("atmosphere.setCloudRangeMax", 0, 1, 0.01, [1]);
      CosmoScout.gui.initSlider("atmosphere.setCloudTypeMin", 0, 1, .01, [0]);
      CosmoScout.gui.initSlider("atmosphere.setCloudTypeMax", 0, 1, .01, [1]);
      CosmoScout.gui.initSlider("atmosphere.setCloudDensityMultiplier", .1, 10, .1, [1]);
      CosmoScout.gui.initSlider("atmosphere.setCloudAbsorption", 0, 1, .01, [0]);
      CosmoScout.gui.initSlider("atmosphere.setCloudCoverageExponent", .1, 5, .1, [1]);
      CosmoScout.gui.initSlider("atmosphere.setCloudCutoff", 0, 1, .01, [.1]);
      CosmoScout.gui.initSlider("atmosphere.setCloudLFRepetitionScale", 100, 50000, 10, [5000]);
      CosmoScout.gui.initSlider("atmosphere.setCloudHFRepetitionScale", 100, 20000, 10, [768]);
    }
  }

  CosmoScout.init(AtmosphereApi);
})();
