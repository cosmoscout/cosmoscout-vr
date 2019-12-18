class StatisticsApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'statistics';

    _values = [];

    _colorHash;

    _minTime = 1000;

    _maxValue = 1e9 / 30;

    _alpha = 0.95;

    constructor() {
      super();

      if (typeof ColorHash !== 'undefined') {
        this._colorHash = new ColorHash({ lightness: 0.5, saturation: 0.3 });
      } else {
        console.error('Class \'ColorHash\' not defined.');
      }
    }

    _resetTimes(data) {
      this._values.forEach((value) => {
        if (typeof data[value.name] !== 'undefined') {
          value.timeGPU = data[value.name][0];
          value.timeCPU = data[value.name][1];
          data[value.name][0] = -1;
          data[value.name][1] = -1;
        } else {
          value.timeGPU = 0;
          value.timeCPU = 0;
        }
      });

      return data;
    }

    _addNewElements(data) {
      for (const key in data) {
        if (!data.hasOwnProperty(key)) {
          continue;
        }

        if (data[key][0] >= 0) {
          this._values.push({
            name: key,
            timeGPU: data[key][0],
            timeCPU: data[key][1],
            avgTimeGPU: data[key][0],
            avgTimeCPU: data[key][1],
            color: this._colorHash.hex(key),
          });
        }
      }
    }

    /**
     *
     * @param data {string}
     * @param frameRate {number}
     */
    setData(data, frameRate) {
      this._values = [];
      data = JSON.parse(data);

      // first set all times to zero
      data = this._resetTimes(data);


      // then add all new elements
      this._addNewElements(data);


      // remove all with very little contribution
      this._values = this._values.filter((element) => element.timeGPU > this._minTime || element.timeCPU > this._minTime
            || element.avgTimeGPU > this._minTime || element.avgTimeCPU > this._minTime);

      // update average values
      this._values.forEach((element) => {
        element.avgTimeGPU = element.avgTimeGPU * this._alpha + element.timeGPU * (1 - this._alpha);
        element.avgTimeCPU = element.avgTimeCPU * this._alpha + element.timeCPU * (1 - this._alpha);
      });


      // sort by average
      this._values.sort((a, b) => (b.avgTimeGPU + b.avgTimeCPU) - (a.avgTimeGPU + a.avgTimeCPU));

      this._insertHtml(frameRate);
    }

    _insertHtml(frameRate) {
      const container = document.getElementById('statistics');
      container.innerHTML = '';

      const maxEntries = Math.min(10, this._values.length);
      const maxWidth = container.offsetWidth;

      container.innerHTML += `<div class="label"><strong>FPS: ${frameRate.toFixed(2)}</strong></div>`;
      /*        for (let i = 0; i < maxEntries; ++i) {
                      const widthGPU = maxWidth * this._values[i].avgTimeGPU / this._maxValue;
                      const widthCPU = maxWidth * this._values[i].avgTimeCPU / this._maxValue;

                      container.innerHTML += `<div class="item">
                      <div class="bar gpu" style="background-color:${this._values[i].color}; width:${widthGPU}px"><div class='label'>gpu: ${(this._values[i].avgTimeGPU * 0.000001).toFixed(1)} ms</div></div>
                      <div class="bar cpu" style="background-color:${this._values[i].color}; width:${widthCPU}px"><div class='label'>cpu: ${(this._values[i].avgTimeCPU * 0.000001).toFixed(1)} ms</div></div>
                      <div class='label'>${this._values[i].name}</div>
                  </div>`;

                  } */
    }
}
