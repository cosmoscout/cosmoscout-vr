////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

(() => {
  /**
   * Atmosphere Api
   */
  class WfsOverlaysApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'wfsOverlays';

    /**
     * @inheritDoc
     */
    init() {
    }
  }

  CosmoScout.init(WfsOverlaysApi);
})();
