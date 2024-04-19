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
    onTourButtonClick(button, tourname) {
      console.log("testing0" + button.checked)

      if (button.checked) {
        let buttons = document.querySelectorAll(".guided-tour-button");
        buttons.forEach(b => {
          if (b != button) {
            b.checked = false;
          }
        })
        console.log("testing1" + button.checked)
        CosmoScout.callbacks.guidedTours.loadTour(tourname);

      }
      else {
        console.log("testing2")
        CosmoScout.callbacks.guidedTours.loadTour("none");
      }
    }
    setProgress(tourName, cpCount, cpVisited)
    {
        console.log(tourName, cpCount, cpVisited);
    }
  }

  CosmoScout.init(GuidedToursApi);
})();
