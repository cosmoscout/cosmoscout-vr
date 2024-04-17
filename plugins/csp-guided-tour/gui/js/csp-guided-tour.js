////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

(() => {
  /**
   * Measurement Tools
   */
  class GuidedToursApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'guidedTours';

    /**
     * TODO
     * @param name {string}
     */
    // eslint-disable-next-line class-methods-use-this
    add(name) {
      const area = document.getElementById('tour-buttons');

      const tourButton = CosmoScout.gui.loadTemplateContent('tour-button-template');

      tourButton.innerHTML = tourButton.innerHTML.replace(/%TOURNAME%/g, name).trim();

      area.appendChild(tourButton);
    }
  }

  CosmoScout.init(GuidedToursApi);
})();
