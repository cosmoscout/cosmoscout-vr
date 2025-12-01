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

    _requestSatellite() {
        const bodyId = (this._nextId++).toString();
        const name = this._nameInput.value;
        console.log(`Requesting satellite "${name}" (${bodyId})`);
        fetch(`${this._spiceServer}/processes/position/execute`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                "id": bodyId,
                "center": 399,
                "frame": "J2000",
                "MU": 398600,
                "SMA": 6733,
                "ECC": 0.03190221275591707,
                "INC": 90.00014746268153,
                "RAAN": 0.004,
                "AOP": 270,
                "M": 0,
                "START": this._startDateDiv.value,
                "END": this._endDateDiv.value,
                "DELTA": 60,
                "description": "This is a test"
            }),
        })
            .then(res => res.json())
            .then(res => {
                const id = res.output.bsp.replace(".bsp", "");
                this._requestedSatellites.push({
                    "bodyId": bodyId,
                    "jobId": id,
                    "name": name,
                });
            });
    }

    _checkProcessStatus(job) {
        fetch(`${this._spiceServer}/jobs/${job.jobId}`)
            .then(res => res.json())
            .then(res => {
                if (res.status == "running") {
                    setTimeout(() => { this._requestedSatellites.push(job); }, 1000);
                } else if (res.status == "submitted") {
                    CosmoScout.callbacks.satellites.addSatellite(job.name, job.bodyId, job.jobId);
                }
            });
    }

    init() {
        // Set up server addresses
        this._renderServer = "http://localhost:9002";
        this._spiceServer = "http://localhost:8000";

        // Set up state
        this._connectionEstablished = false;
        this._needImage = true;
        this._requestedSatellites = [];
        this._nextId = -11111;

        // Init/Get various DOM elements
        this._viewDiv = CosmoScout.gui.loadTemplateContent('satellite-view-template');
        document.getElementById('cosmoscout').appendChild(this._viewDiv);

        this._viewCanvas = document.getElementById('satellite-view-canvas');
        this._viewCtx = this._viewCanvas.getContext("2d");

        this._nameInput = document.getElementById("satellite-add-name");
        this._startDateDiv = document.getElementById("satellite-add-start-date");
        this._endDateDiv = document.getElementById("satellite-add-end-date");

        document.querySelector("#satellite-add-start-date + div > button").onclick = () => {
            this._startDateDiv.value =
                CosmoScout.state.simulationTime.toISOString().replace('T', ' ').slice(0, 19);
        };
        document.querySelector("#satellite-add-end-date + div > button").onclick = () => {
            this._endDateDiv.value =
                CosmoScout.state.simulationTime.toISOString().replace('T', ' ').slice(0, 19);
        };

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
        this._requestedSatellites.forEach(job => this._checkProcessStatus(job));
        this._requestedSatellites = [];
    }
  }

  CosmoScout.init(SatellitesApi);
})();
