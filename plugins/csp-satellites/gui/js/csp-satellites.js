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
            body: 'CosmoScout.callbacks.navigation.setBodyFull("-10001", "VLEO_OFFSET", 0.1, 0.1, 0.1, 0, 1, 0, 0, 0)'
        })
            .then(res => {
                if (res.ok) {
                    this._connectionEstablished = true;
                }
            });
    }

    _resetDate() {
        const time = CosmoScout.timeline._centerTime.toISOString();
        fetch(`${this._renderServer}/run-js`, {
            method: "POST",
            body: `CosmoScout.callbacks.time.setDate("${time}")`
        });
    }

    _fetchImage() {
        this._resetDate();

        const params = new URLSearchParams();
        params.append("width", "200");
        params.append("height", "200");
        params.append("gui", "false");
        params.append("delay", "0");
        params.append("format", "png");
        fetch(`${this._renderServer}/capture?${params}`)
            .then(res => res.blob())
            .then(blob => createImageBitmap(blob))
            .then(image => {
                this._viewCtx.drawImage(image, 0, 0);
                this._needImage = true;
            })
            .catch(e => console.error(`Error fetching satellite view: ${e}`));
    }

    init() {
        this._connectionEstablished = false;
        this._needImage = true;
        this._viewDiv = CosmoScout.gui.loadTemplateContent('satellite-view-template');
        document.getElementById('cosmoscout').appendChild(this._viewDiv);
        this._viewCanvas = document.getElementById('satellite-view-canvas');
        this._viewCtx = this._viewCanvas.getContext("2d");
        this._renderServer = "http://localhost:9002";

        this._resetObserver();
    }

    deinit() {
        document.getElementById('cosmoscout').removeChild(this._viewDiv);
    }

    update() {
        if (this._connectionEstablished && this._needImage) {
            this._needImage = false;
            this._fetchImage();
        }
    }
  }

  CosmoScout.init(SatellitesApi);
})();
