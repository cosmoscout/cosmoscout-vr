////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

(() => {
  /**
   * Satellite Api
   */
  class SatellitesApi extends IApi {
    /**
     * @inheritDoc
     * @type {string}
     */
    name = 'satellite';

    _resetObserver() {
        fetch(`${this._renderServer}/run-js`, {
            method: "POST",
            body: 'CosmoScout.callbacks.navigation.setBodyFull("-10001", "VLEO_PARAM", 0, 0, 0, 0.707, 0, 0, 0.707, 0)'
        })
            .then(res => {
                if (res.ok) {
                    this._connectionEstablished = true;
                }
            });
    }

    _resetDate() {
        const time = CosmoScout.timeline._centerTime.toISOString().replace('T', ' ').slice(0, 19);
        fetch(`${this._renderServer}/run-js`, {
            method: "POST",
            body: `CosmoScout.callbacks.time.setDate("${time}")`
        });
    }

    _fetchImage() {
        const width = 200;
        const height = 200;
        const delay = 10;
        //this._viewImg.src = `${this._renderServer}/capture?width=${width}&height=${height}&delay=${delay}&gui=false#${new Date().getTime()}`;

        const params = new URLSearchParams();
        params.append("width", "600");
        params.append("height", "600");
        params.append("gui", "false");
        params.append("delay", "10");
        params.append("format", "jpeg");
        fetch(`${this._renderServer}/capture?${params}`)
            .then(res => res.blob())
            .then(blob => {
                console.log(blob.size);
                console.log(blob.type);
                const imgData = URL.createObjectURL(blob);
                //this._viewImg.src = imgData;
            })
            .catch(e => console.error(`Error fetching satellite view: ${e}`));
    }

    init() {
        this._connectionEstablished = false;
        this._viewDiv = CosmoScout.gui.loadTemplateContent('satellite-view-template');
        document.getElementById('cosmoscout').appendChild(this._viewDiv);
        this._viewImg = document.querySelector('#satellite-view-img');
        this._renderServer = "http://localhost:9002";

        this._resetObserver();

        this._viewImg.addEventListener("load", () => {
            this._fetchImage();
        });
    }

    deinit() {
        document.getElementById('cosmoscout').removeChild(this._viewDiv);
    }

    update() {
    }
  }

  CosmoScout.init(SatellitesApi);
})();
