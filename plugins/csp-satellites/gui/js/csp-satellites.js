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
                    this._state = this._states["idle"];
                }
            })
            .catch(e => console.error(`Error setting body: ${e}`));
    }

    _resetDate() {
        const time = CosmoScout.timeline._centerTime;
        fetch(`${this._renderServer}/run-js`, {
            method: "POST",
            body: `CosmoScout.callbacks.time.setDate("${time.toISOString()}")`
        })
            .catch(e => console.error(`Error setting date: ${e}`));
        this._lastImageTime = time;
        this._needImage = true;
    }

    _setFieldOfView(deg) {
        const rad = deg / 180 * Math.PI;
        const sensorDiagonal = 42;
        const focalLength = sensorDiagonal / 2 / Math.tan(rad / 2);
        fetch(`${this._renderServer}/run-js`, {
            method: "POST",
            body: `CosmoScout.callbacks.graphics.setFocalLength(${focalLength})`
        })
            .catch(e => console.error(`Error setting field of view: ${e}`));
        this._needImage = true;
    }

    _checkShips(imageBlob) {
        const data = new FormData();
        data.append("file", imageBlob);
        fetch(`${this._shipServer}/processes/ships/execute`, {
            method: "POST",
            body: data,
        }).then(res => res.json())
            .then(res => {
                res.output.json[0].matches.forEach(match => {
                    this._viewCtx.strokeRect(match[0] - match[2]/2, match[1] - match[3]/2, match[2], match[3]);
                });
                this._state = this._states["idle"];
            })
            .catch(e => console.error(`Error checking for ship: ${e}`));
    }

    _fetchImage() {
        const params = new URLSearchParams();
        params.append("width", "320");
        params.append("height", "320");
        params.append("gui", "false");
        params.append("delay", "0");
        params.append("format", "png");
        fetch(`${this._renderServer}/capture?${params}`)
            .then(res => res.blob())
            .then(blob => {
                this._state = this._states["awaitShips"];
                createImageBitmap(blob).then(image => {
                    this._viewCtx.drawImage(image, 0, 0);
                });
                this._checkShips(blob);
            })
            .catch(e => console.error(`Error fetching satellite view: ${e}`));
    }

    _requestSatellite() {
        const bodyId = (this._nextId++).toString();
        const name = this._nameInput.value;
        const params = {
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
        };

        console.log(`Requesting satellite "${name}" (${bodyId})`);
        const promisedPosition = fetch(`${this._spiceServer}/processes/position/execute`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(params),
        }).then(res => res.json());
        const promisedOrientation = fetch(`${this._spiceServer}/processes/orientation/execute`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(params),
        }).then(res => res.json());
        Promise.all([promisedPosition, promisedOrientation])
            .then(res => {
                const [resPos, resOrient] = res;
                const id = resPos.output.bsp.replace(".bsp", "");
                this._requestedSatellites.push({
                    "bodyId": bodyId,
                    "jobId": id,
                    "bodyName": name,
                    "existenceStart": this._startDateDiv.value,
                    "existenceEnd": this._endDateDiv.value,
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
                    CosmoScout.callbacks.satellites.addSatellite(JSON.stringify(job));
                }
            });
    }

    init() {
        // State enums
        this._states = {
            "connecting": 0,
            "idle": 1,
            "awaitImage": 2,
            "awaitShips": 3,
        };

        // Set up server addresses
        this._renderServer = "http://localhost:9002";
        this._spiceServer = "http://localhost:8000";
        this._shipServer = "http://localhost:8001";

        // Set up state
        this._state = this._states["connecting"];
        this._needImage = true;
        this._requestedSatellites = [];
        this._nextId = -11111;

        // Init/Get various DOM elements
        this._viewDiv = CosmoScout.gui.loadTemplateContent('satellite-view-template');
        document.getElementById('cosmoscout').appendChild(this._viewDiv);

        this._viewCanvas = document.getElementById('satellite-view-canvas');
        this._viewCtx = this._viewCanvas.getContext("2d");
        this._viewCtx.strokeStyle = "red";

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
        // Set render server time if it changed since last image
        if (this._state === this._states["idle"] && this._lastImageTime != CosmoScout.timeline._centerTime) {
            this._resetDate();
        }
        if (this._state === this._states["idle"] && this._needImage) {
            this._state = this._states["awaitImage"];
            this._needImage = false;
            this._fetchImage();
        }
        this._requestedSatellites.forEach(job => this._checkProcessStatus(job));
        this._requestedSatellites = [];
    }
  }

  CosmoScout.init(SatellitesApi);
})();
