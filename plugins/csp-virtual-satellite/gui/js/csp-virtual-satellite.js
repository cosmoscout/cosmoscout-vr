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
      this._repoSelect = document.querySelector(`[data-callback="virtualSatellite.setRepository"]`);
    }

    /**
     * @inheritDoc
     */
    update() {
    }

    authenticate() {
      const username = document.querySelector('#virtualSatellite-username').value;
      const password = document.querySelector('#virtualSatellite-password').value;

      CosmoScout.callbacks.virtualSatellite.authenticate(username, password);
    }

    resetRepoSelect() {
      $(this._repoSelect).selectpicker("destroy");
      $(this._repoSelect).selectpicker();
      CosmoScout.gui.clearDropdown("virtualSatellite.setRepository");
      CosmoScout.gui.addDropdownValue("virtualSatellite.setRepository", "None", "None", true);
      document.querySelector("#virtualSatellite-repo-root").hidden = true;
    }

    refreshRepoSelect() {
      $(this._repoSelect).selectpicker("refresh");
      document.querySelector("#virtualSatellite-repo-root").hidden = false;
    }


    addRepo(repo) {
      const option = document.createElement('option');

      option.value     = repo;
      option.selected  = false;
      option.disabled  = false;
      option.innerHTML = repo;

      this._repoSelect.appendChild(option);
    }

    /**
     * Recreate the dropdown for SEI selection
     */
    resetSEISelect() {
      $(this._rootSEISelect).selectpicker("destroy");
      $(this._rootSEISelect).selectpicker();
      CosmoScout.gui.clearDropdown("virtualSatellite.setRootSEI");
      CosmoScout.gui.addDropdownValue("virtualSatellite.setRootSEI", "None", "None", true);
      document.querySelector("#virtualSatellite-sei-root").hidden = true;
    }

    /**
     * Call the refresh function on the SEI dropdown.
     */
    refreshSEISelect() {
      $(this._rootSEISelect).selectpicker("refresh");
      document.querySelector("#virtualSatellite-sei-root").hidden = false;
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
