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
      this._colorDiv = document.getElementById("bookmark-editor-color");
    }

    setFeatureProperties (properties) {
      const props = JSON.parse(properties);
      const tableContainer = document.getElementById("tableContainer");
      const tbody = document.getElementById("tableBody");

      tbody.innerHTML = " ";

      props.featureTypes[0].properties.forEach((item) => {

        const row = document.createElement("tr");

        const nameCell = document.createElement("td");
        nameCell.textContent = item.name;
        row.appendChild(nameCell);

        const typeCell = document.createElement("td");
        typeCell.textContent = item.localType;
        row.appendChild(typeCell);

        const radioButton = document.createElement("input");
        radioButton.type = "radio";
        radioButton.name = "selectedColor";
        radioButton.value = item.name;
        radioButton.style.display = "block";
        
        radioButton.addEventListener("click", () => { CosmoScout.callbacks.wfsOverlays.setColor(item.name)});
        const colorCell = document.createElement("td");
        colorCell.appendChild(radioButton);
        row.appendChild(colorCell);

        const radioButtonTime = document.createElement("input");
        radioButtonTime.type = "radio";
        radioButtonTime.name = "selectedTime";
        radioButtonTime.value = item.name;
        radioButtonTime.style.display = "block";

        const timeCell = document.createElement("td");
        timeCell.appendChild(radioButtonTime);
        row.appendChild(timeCell);

        /* 
        Object.values(item).forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.appendChild(cell);
        });
        */
        tbody.appendChild(row);
      });
    }
  }

  CosmoScout.init(WfsOverlaysApi);
})();
