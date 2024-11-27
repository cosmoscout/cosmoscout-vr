////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

(() => {
  /**
  * Virtual Satellite Api
  */
  class VirtualSatelliteApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'virtualSatellite'

    /**
     * @inheritDoc
     */
    init() {
      this._rootSEISelect = document.querySelector(`[data-callback="virtualSatellite.setRootSEI"]`);
    }

    /**
     * @inheritDoc
     */
    update() {
    }

    /**
     * Recreate the dropdown for SEI selection
     */
    resetSEISelect() {
      $(this._rootSEISelect).selectpicker("destroy");
      $(this._rootSEISelect).selectpicker();
      CosmoScout.gui.clearDropdown("virtualSatellite.setRootSEI");
      CosmoScout.gui.addDropdownValue("virtualSatellite.setRootSEI", "None", "None", true);
    }

    /**
     * Call the refresh function on the SEI dropdown.
     */
    refreshSEISelect() {
      $(this._rootSEISelect).selectpicker("refresh");
    }

    /**
     * Add a new SEI to the SEI dropdown.
     * Indent it according to the SEI hierarchy.
     *
     * @param {string} name The internal name of the SEI
     * @param {string} title The human-readable title of the SEI
     * @param {boolean} active Whether the SEI is currently selected
     */
    addSEI(name, title, active) {
      const option = document.createElement('option');

      option.value     = name;
      option.selected  = active;
      option.disabled  = false;
      option.innerHTML = title;

      this._rootSEISelect.appendChild(option);
    }
  }

  CosmoScout.init(VirtualSatelliteApi);
})();

//# sourceMappingURL=js/csp-virtual-satellite.js
