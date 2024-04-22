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

      if (button.checked) {
        let buttons = document.querySelectorAll(".guided-tour-button");
        buttons.forEach(b => {
          if (b != button) {
            b.checked = false;
          }
        })
        CosmoScout.callbacks.guidedTours.loadTour(tourname);

      }
      else {
        CosmoScout.callbacks.guidedTours.loadTour("none");
      }
    }
    resetAll() {
      let buttons = document.querySelectorAll(".guided-tour-button");
      buttons.forEach(b => {
          b.checked = false;
          const tourStatusLabel = b.nextElementSibling.querySelector('.guided-tour-status');
          const label = tourStatusLabel.innerText.trim(); 
          if (label !== "") {
              const [firstNumber, secondNumber] = label.split('/');
              const newFirstNumber = "0";
              const newLabel = `${newFirstNumber}/${secondNumber}`;
              tourStatusLabel.innerText = newLabel;
          }
      });
  }
  
    setProgress(tourName, cpCount, cpVisited) {

      console.log(tourName, cpCount, cpVisited);
      const tourButtons = document.querySelectorAll('.guided-tour-label');

      tourButtons.forEach(button => {
        if (tourName == button.querySelector('span').innerText) {
          const tourStatusLabel = button.querySelector('.guided-tour-status');
          tourStatusLabel.innerText = cpVisited + "/" + cpCount;

        }
      });

    }
  }


  CosmoScout.init(GuidedToursApi);
})();
