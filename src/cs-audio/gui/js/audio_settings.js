////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

(() => {
    /**
     * Audio Api
     */
    class AudioApi extends IApi {
      /**
       * @inheritDoc
       */
      name = 'audio';
  
      /**
       * @inheritDoc
       */
      init() {
        CosmoScout.gui.initSlider("audio.masterVolume", 0.0, 5, 0.05, [1]);
      }
    }
  
    CosmoScout.init(AudioApi);
  })();
  