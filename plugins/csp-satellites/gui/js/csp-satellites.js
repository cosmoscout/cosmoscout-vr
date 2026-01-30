////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

(() => {
  const STATES = {
    "connecting": 0,
    "idle": 1,
    "awaitImage": 2,
    "awaitShips": 3,
  };

  /**
   * Base class for managing a video + ship detection stream.
   * Mainly takes care of creating a context and drawing to it.
   */
  class SatelliteStream {
    constructor(canvas) {
        this._drawCtx = canvas.getContext("2d");
        this._drawCtx.strokeStyle = "red";
    }

    drawImg(blob) {
        return new Promise((resolve, reject) => {
            createImageBitmap(blob).then((img) => {
                this._drawCtx.drawImage(img, 0, 0);
                resolve();
            });
        });
    }

    drawRect(bbox, centered) {
        if (centered) {
            this._drawCtx.strokeRect(bbox[0] - bbox[2]/2, bbox[1] - bbox[3]/2, bbox[2], bbox[3]);
        } else {
            this._drawCtx.strokeRect(bbox[0], bbox[1], bbox[2], bbox[3]);
        }
    }
  }

  /**
   * Class managing a video + ship detection stream that is pushed from some remote server.
   * I.e. the remote server continuously sends messages to us and we have to handle them as we get them.
   */
  class PushedStream extends SatelliteStream {
    state = STATES["connecting"];

    constructor(canvas) {
        super(canvas);
    }
  }

  /**
   * Class managing a video + ship detection stream that is pulled from some remote server.
   * I.e. we are responsible for requesting a new image whenever necessary.
   */
  class PulledStream extends SatelliteStream {
    state = STATES["connecting"];

    constructor(canvas) {
        super(canvas);
    }

    pull() {

    }
  }

  /**
   * Class for managing a draggable window displaying a satellite view.
   */
  class ViewWindow {
    constructor() {
        this.window = CosmoScout.gui.loadTemplateContent("satellite-view-template");
        document.getElementById("cosmoscout").appendChild(this.window);

        this.canvas   = this.window.getElementsByClassName("satellite-view-canvas")[0];
        this.controls = this.window.getElementsByClassName("satellite-view-controls")[0];
        this.slider   = this.window.getElementsByClassName("satellite-view-fov")[0];

        // Init the slider
        noUiSlider.create(this.slider, {
            start: [1],
            range: {
                "min": [0],
                "max": [7]
            },
            step: 0.1
        });
    }

    hideControls() {
        this.controls.style.display = "none"
    }

    showControls() {
        this.controls.style.display = "block"
    }
  }

  /**
   * Class managing the connection to and synchronization with a CosmoScout render server.
   */
  class RenderServer {
    constructor(url) {
        this._url = url;
    }

    /**
     * Get a current image (as Blob) from the render server.
     */
    getImage() {
        const params = new URLSearchParams();
        params.append("width", "320");
        params.append("height", "320");
        params.append("gui", "false");
        params.append("delay", "4");
        params.append("format", "png");
        return fetch(`${this._url}/capture?${params}`)
            .then(res => res.blob());
    }

    /**
     * Creates a default promise for fetches that don't return any meaningful results.
     * Also makes sure that no file descriptors are leaked.
     */
    _handleEmptyResponse(fetchPromise, errorText) {
        return new Promise((resolve, reject) => {
            fetchPromise
                .then(res => {
                    if (res.ok) {
                        resolve();
                    } else {
                        reject(`${errorText}: ${res}`);
                    }
                    // Apparently our current CEF version leaks file descriptors for each fetch,
                    // unless the body of the response is handled in some way.
                    // Because of this we convert it to blob here and then just drop the response.
                    return res.blob();
                })
                .catch(e => reject(`${errorText}: ${e}`));
        });
    }

    /**
     * Reset the remote observer position to its default frame and position.
     */
    resetLocation() {
        const promise = fetch(`${this._url}/run-js`, {
            method: "POST",
            body: 'CosmoScout.callbacks.navigation.setBodyFull("-10001", "VLEO_OFFSET", 0.1, 0.1, 0.1, 0, 1, 0, 0, 0)'
        });
        return this._handleEmptyResponse(promise, "Error setting observer location");
    }

    /**
     * Sync the remote time to our time.
     */
    syncTime() {
        const time = CosmoScout.timeline._centerTime;
        return this.callSetter("time.setDate", `"${time.toISOString()}"`);
    }

    /**
     * Set an arbitrary remote parameter to the given time value.
     */
    callSetter(parameter, value) {
        const promise = fetch(`${this._url}/run-js`, {
            method: "POST",
            body: `CosmoScout.callbacks.${parameter}(${value})`
        });
        return this._handleEmptyResponse(promise, `Error setting ${parameter}`);
    }
  }

  /**
   * Satellite Api
   */
  class SatellitesApi extends IApi {
    /**
     * @inheritDoc
     * @type {string}
     */
    name = 'satellites';

    /**
     * Externally used functions for push-style video stream.
     */
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
        this._realStream.drawImg(blob).then(() => {
            bboxs.forEach(bbox => {
                this._realStream.drawRect(bbox, false);
            });
        });
    }

    /**
     * Internally used functions.
     */
    setFieldOfView(satellite, deg, emitCallback=true) {
        const rad = deg / 180 * Math.PI;
        const sensorDiagonal = 42;
        const focalLength = sensorDiagonal / 2 / Math.tan(rad / 2);
        if (satellite === this._activeSatellite) {
            this._renderServer.callSetter("graphics.setFocalLength", focalLength);
            this._needImage = true;
        }
        if (emitCallback) {
            CosmoScout.callbacks.satellites.setFieldOfView(satellite, deg);
        } else {
            this._virtView.slider.noUiSlider.set([deg], false);
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
                    this._virtStream.drawRect(match, true);
                });
                this._state = this._states["idle"];
            })
            .catch(e => {
                console.error(`Error checking for ship: ${e}`);
                this._state = this._states["idle"];
            });
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
        this._renderServerUrl = `http://${hostA}:9002`;
        this._spiceServer = `http://${hostA}:8000`;
        this._shipServer = `http://${hostA}:8001`;

        // Set up state
        this._renderServer = new RenderServer(this._renderServerUrl);
        this._state = this._states["connecting"];
        this._needImage = true;
        this._requestedSatellites = [];
        this._nextId = -11111;
        this._activeSatellite = "VLEO";
        this._satellites = {};

        // Init/Get various DOM elements
        this._virtView = new ViewWindow();
        this._realView = new ViewWindow();
        this._realView.hideControls();
        this._virtStream = new PulledStream(this._virtView.canvas);
        this._realStream = new PushedStream(this._realView.canvas);

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

        this._virtView.slider.noUiSlider.on('slide', (values, handle, unencoded) => {
            CosmoScout.satellites.setFieldOfView(this._activeSatellite, parseFloat(unencoded));
        });

        this._renderServer.resetLocation()
            .then(() => this._state = this._states["idle"])
            .catch((e) => console.error(e));
    }

    deinit() {
        document.getElementById('cosmoscout').removeChild(this._viewDiv);
    }

    update() {
        // Set render server time if it changed since last image
        if (this._state === this._states["idle"] && this._lastImageTime != CosmoScout.timeline._centerTime) {
            this._lastImageTime = CosmoScout.timeline._centerTime;
            this._renderServer.syncTime();
            this._needImage = true;
                //.then(() => this._needImage = true)
                //.catch((e) => console.error(e));
        }
        if (this._state === this._states["idle"] && this._needImage) {
            this._state = this._states["awaitImage"];
            this._needImage = false;
            this._renderServer.getImage()
                .then((blob) => {
                    this._state = this._states["awaitShips"];
                    this._virtStream.drawImg(blob);
                    this._checkShips(blob);
                })
                .catch(e => console.error(`Error fetching satellite view: ${e}`));
        }
        this._requestedSatellites.forEach(job => this._checkProcessStatus(job));
        this._requestedSatellites = [];
    }
  }

  CosmoScout.init(SatellitesApi);
})();
