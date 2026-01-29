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
    name = 'satellites';

    setSatelliteConfiguration(id) {
        const models = [
            "../share/resources/models/VLEO_centered.glb",
            "../share/resources/models/VLEO_alt.glb",
        ];
        CosmoScout.callbacks.satellites.setSatelliteModel(models[id]);
    }

    updateDetection(imgB64, bboxs, inference_speed) {
        const imgBytes = Uint8Array.from(atob(imgB64), c => c.charCodeAt(0));
        const blob = new Blob([imgBytes], {type: "image/jpeg"});
        createImageBitmap(blob).then((img) => {
            this._viewCtx.drawImage(img, 0, 0);
            bboxs.forEach(bbox => {
                this._viewCtx.strokeRect(bbox[0], bbox[1], bbox[2], bbox[3]);
            });
        });
    }

    _resetObserver() {
        fetch(`${this._renderServer}/run-js`, {
            method: "POST",
            body: 'CosmoScout.callbacks.navigation.setBodyFull("-10001", "VLEO_OFFSET", 0.1, 0.1, 0.1, 0, 1, 0, 0, 0)'
        })
            .then(res => {
                if (res.ok) {
                    this._state = this._states["idle"];
                }
                // [1] Apparently our current CEF version leaks file descriptors for each fetch,
                // unless the body of the response is handled in some way.
                // Because of this we convert it to blob here and then just drop the response.
                return res.blob();
            })
            .catch(e => console.error(`Error setting body: ${e}`));
    }

    _resetDate() {
        const time = CosmoScout.timeline._centerTime;
        fetch(`${this._renderServer}/run-js`, {
            method: "POST",
            body: `CosmoScout.callbacks.time.setDate("${time.toISOString()}")`
        })
            .then(res => res.blob()) // See [1]
            .catch(e => console.error(`Error setting date: ${e}`));
        this._lastImageTime = time;
        this._needImage = true;
    }

    setFieldOfView(satellite, deg, emitCallback=true) {
        const rad = deg / 180 * Math.PI;
        const sensorDiagonal = 42;
        const focalLength = sensorDiagonal / 2 / Math.tan(rad / 2);
        if (satellite === this._activeSatellite) {
            fetch(`${this._renderServer}/run-js`, {
                method: "POST",
                body: `CosmoScout.callbacks.graphics.setFocalLength(${focalLength})`
            })
                .then(res => res.blob()) // See [1]
                .catch(e => console.error(`Error setting field of view: ${e}`));
            this._needImage = true;
        }
        if (emitCallback) {
            CosmoScout.callbacks.satellites.setFieldOfView(satellite, deg);
        } else {
            this._fovSlider.noUiSlider.set([deg], false);
        }
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
            .catch(e => {
                            console.error(`Error checking for ship: ${e}`);
                            this._state = this._states["idle"];
                        });
    }

    _fetchImage() {
        const params = new URLSearchParams();
        params.append("width", "320");
        params.append("height", "320");
        params.append("gui", "false");
        params.append("delay", "4");
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

    _getBodyIdAndName() {
        let bodyId;
        let name;
        if (this._inputs["satellite-id"].value == 0) {
            bodyId = (this._nextId++).toString();
            name = this._inputs.name.value;
        } else {
            bodyId = this._inputs["satellite-id"].value;
            name = this._satellites[bodyId].name;
        }
        return [bodyId, name];
    }

    _requestSatellite() {
        const [bodyId, name] = this._getBodyIdAndName();
        const params = {
            "id": bodyId,
            "center": 399,
            "frame": "J2000",
            "MU": 398600,
            "SMA": parseFloat(this._inputs.sma.value),
            "ECC": parseFloat(this._inputs.ecc.value),
            "INC": parseFloat(this._inputs.inc.value),
            "RAAN": parseFloat(this._inputs.raan.value),
            "AOP": parseFloat(this._inputs.aop.value),
            "M": parseFloat(this._inputs.m.value),
            "START": this._inputs["start-date"].value,
            "END": this._inputs["end-date"].value,
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
        /*const promisedOrientation = fetch(`${this._spiceServer}/processes/orientation/execute`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(params),
        }).then(res => res.json());*/
        Promise.all([promisedPosition/*, promisedOrientation*/])
            .then(res => {
                const [resPos, resOrient] = res;
                const id = resPos.output.bsp.replace(".bsp", "");
                this._requestedSatellites.push({
                    "bodyId": bodyId,
                    "jobId": id,
                    "bodyName": name,
                    "existenceStart": this._inputs["start-date"].value,
                    "existenceEnd": this._inputs["end-date"].value,
                });
            })
            .catch(e => console.error(`Error requesting satellite: ${e}`));
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

    addSatellite(name, center, frame) {
        this._satellites[center] = {
            "name": name,
            "frame": frame,
        };
        CosmoScout.gui.addDropdownValue("satellites.setSatellite", center, name, false);
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
        const hostA = "localhost";
        const hostB = "129.247.51.9";
        const hostC = "129.247.51.78";
        this._renderServer = `http://${hostA}:9002`;
        this._spiceServer = `http://${hostA}:8000`;
        this._shipServer = `http://${hostA}:8001`;

        // Set up state
        this._state = this._states["connecting"];
        this._needImage = true;
        this._requestedSatellites = [];
        this._nextId = -11111;
        this._activeSatellite = "VLEO";
        this._satellites = {};

        // Init/Get various DOM elements
        this._viewDiv = CosmoScout.gui.loadTemplateContent('satellite-view-template');
        document.getElementById('cosmoscout').appendChild(this._viewDiv);

        this._viewCanvas = document.getElementById('satellite-view-canvas');
        this._viewCtx = this._viewCanvas.getContext("2d");
        this._viewCtx.strokeStyle = "red";

        this._inputs = {};
        for (const input of ["name", "start-date", "end-date", "sma", "ecc", "inc", "raan", "aop", "m"]) {
            this._inputs[input] = document.getElementById(`satellite-add-${input}`);
        }
        this._inputs["satellite-id"] = document.querySelector(`[data-callback="satellites.setSatellite"]`);

        document.querySelector("#satellite-add-start-date + div > button").onclick = () => {
            this._inputs["start-date"].value =
                CosmoScout.state.simulationTime.toISOString().replace('T', ' ').slice(0, 19);
        };
        document.querySelector("#satellite-add-end-date + div > button").onclick = () => {
            this._inputs["end-date"].value =
                CosmoScout.state.simulationTime.toISOString().replace('T', ' ').slice(0, 19);
        };

         // Init the slider
        this._fovSlider = document.getElementById("satellite-view-fov");
        noUiSlider.create(this._fovSlider, {
            start: [1],
            range: {
                'min': [0],
                'max': [7]
            },
            step: 0.1
        });

        this._fovSlider.noUiSlider.on('slide', (values, handle, unencoded) => {
            CosmoScout.satellites.setFieldOfView(this._activeSatellite, parseFloat(unencoded));
        });

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
