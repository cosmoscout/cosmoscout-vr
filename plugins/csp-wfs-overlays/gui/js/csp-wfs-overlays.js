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

    // Here we manage the properties display at the GUI
    //-------------------------------------------------
    setFeatureProperties (properties) {

      const props = JSON.parse(properties);
      const tableContainer = document.getElementById("tableContainer");
      const tbody = document.getElementById("tableBody");

      tbody.innerHTML = " ";

      props.featureTypes[0].properties.forEach((item) => {

        const row = document.createElement("tr");

        // name cell
        const nameCell = document.createElement("td");
        nameCell.textContent = item.name;
        row.appendChild(nameCell);

        // type cell
        const typeCell = document.createElement("td");
        typeCell.textContent = item.localType;
        row.appendChild(typeCell);
        
        // color cell
        const colorCell = document.createElement("td");
        if (item.localType == "string") {
          const radioButton = document.createElement("input");
          radioButton.type = "radio";
          radioButton.name = "selectedColor";
          radioButton.value = item.name;
          radioButton.style.display = "block";
          radioButton.addEventListener("click", () => { CosmoScout.callbacks.wfsOverlays.setColor(item.name)});
          colorCell.appendChild(radioButton);
        }
        row.appendChild(colorCell);

        // time cell  (note that unlike for the color cell, we do not process dates in the .cpp)
        const timeCell = document.createElement("td");
        if (item.localType == "date-time") {
          const radioButtonTime = document.createElement("input");
          radioButtonTime.type = "radio";
          radioButtonTime.name = "selectedTime";
          radioButtonTime.value = item.name;
          radioButtonTime.style.display = "block";
          // we could add an event listener here
          timeCell.appendChild(radioButtonTime);  
        }
        row.appendChild(timeCell);

        // append the whole row containing all the cells above (name, type, color, time)
        tbody.appendChild(row);
      });
    }

    // Here we manage the point size selection displayed at the GUI
    //-------------------------------------------------------------
    setSize () {
      const sizeSlider = document.getElementById("size-slider");
      sizeSlider.addEventListener('change', function () {
        const selectedValue = sizeSlider.value;
        CosmoScout.callbacks.wfsOverlays.setSize(selectedValue);
      });
    }

    // Here we manage the line width selection displayed at the GUI
    //-------------------------------------------------------------
    setWidth () {
      const widthSlider = document.getElementById("width-slider");
      widthSlider.addEventListener('change', function () {
        const selectedValue = widthSlider.value;
        CosmoScout.callbacks.wfsOverlays.setWidth(selectedValue);
        // console.log("js::Width transmitido al cpp:", typeof selectedValue);
      });
    }

  }

  CosmoScout.init(WfsOverlaysApi);
})();
