/* global IApi, CosmoScout, ColorHash */

/**
 * Timings Api
 */
class TimingsApi extends IApi {
  /**
   * @inheritDoc
   */
  name = 'timings';

  /**
   * Timing values
   *
   * @type {*[]}
   * @private
   */
  _values = [];

  /**
   * Parsed json
   *
   * @type {Array}
   * @private
   */
  _data;

  /**
   * ColorHash object
   *
   * @type {ColorHash}
   * @private
   */
  _colorHash;

  /**
   * Min time to be used for calculations
   *
   * @type {number}
   * @private
   */
  _minTime = 1000;

  /**
   * @type {number}
   * @private
   */
  _maxValue = 1e9 / 30;

  /**
   * @type {number}
   * @private
   */
  _alpha = 0.95;

  init() {
    if (typeof ColorHash !== 'undefined') {
      this._colorHash = new ColorHash({lightness: 0.5, saturation: 0.3});
    } else {
      console.error('Class \'ColorHash\' not defined.');
    }
  }

  /**
   *
   * @param data {string}
   * @param frameRate {number}
   */
  setData(data, frameRate) {
    this._data = JSON.parse(data);

    // first set all times to zero
    this._resetTimes();

    // then add all new elements
    this._addNewElements();

    // remove all with very little contribution
    const minTime = (element) =>
        element.timeGPU > this._minTime || element.timeCPU > this._minTime ||
        element.avgTimeGPU > this._minTime || element.avgTimeCPU > this._minTime;
    this._values = this._values.filter(minTime);

    // update average values
    this._values.forEach((element) => {
      element.avgTimeGPU = element.avgTimeGPU * this._alpha + element.timeGPU * (1 - this._alpha);
      element.avgTimeCPU = element.avgTimeCPU * this._alpha + element.timeCPU * (1 - this._alpha);
    });

    // sort by average
    this._values.sort((a, b) => (b.avgTimeGPU + b.avgTimeCPU) - (a.avgTimeGPU + a.avgTimeCPU));

    this._insertHtml(frameRate);
  }

  /**
   * Reset times
   *
   * @see {_data}
   * @private
   */
  _resetTimes() {
    this._values.forEach((value) => {
      if (typeof this._data[value.name] !== 'undefined') {
        [value.timeGPU, value.timeCPU] = this._data[value.name];

        this._data[value.name][0] = -1;
        this._data[value.name][1] = -1;
      } else {
        value.timeGPU = 0;
        value.timeCPU = 0;
      }
    });
  }

  /**
   * Add elements to _values
   *
   * @see {_values}
   * @private
   */
  _addNewElements() {
    Object.keys(this._data).forEach((key) => {
      if (this._data[key][0] >= 0) {
        this._values.push({
          name: key,
          timeGPU: this._data[key][0],
          timeCPU: this._data[key][1],
          avgTimeGPU: this._data[key][0],
          avgTimeCPU: this._data[key][1],
          color: this._colorHash.hex(key),
        });
      }
    });
  }

  /**
   * Insert the actual html
   *
   * @param frameRate {number}
   * @private
   */
  _insertHtml(frameRate) {
    const container = document.getElementById('timings');
    CosmoScout.gui.clearHtml(container);

    const maxEntries = Math.min(10, this._values.length);
    const maxWidth   = container.offsetWidth;

    const item = document.createElement('template');

    item.innerHTML = `<div class="label"><strong>FPS: ${frameRate.toFixed(2)}</strong></div>`;

    container.appendChild(item.content);

    for (let i = 0; i < maxEntries; ++i) {
      /* eslint-disable no-mixed-operators */
      const widthGPU = maxWidth * this._values[i].avgTimeGPU / this._maxValue;
      const widthCPU = maxWidth * this._values[i].avgTimeCPU / this._maxValue;
      /* eslint-enable no-mixed-operators */

      item.innerHTML += `<div class="timings-item">
        <div class="bar gpu" style="background-color:${this._values[i].color}; width:${
          widthGPU}px"><div class="label">gpu: ${
          (this._values[i].avgTimeGPU * 0.000001).toFixed(1)} ms</div></div>
        <div class="bar cpu" style="background-color:${this._values[i].color}; width:${
          widthCPU}px"><div class="label">cpu: ${
          (this._values[i].avgTimeCPU * 0.000001).toFixed(1)} ms</div></div>
        <div class="label">${this._values[i].name}</div>
      </div>`;

      container.appendChild(item.content);
    }
  }
}
