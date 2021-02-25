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
   * ColorHash object
   *
   * @type {ColorHash}
   * @private
   */
  _colorHash;

  init() {
    if (typeof ColorHash !== 'undefined') {
      this._colorHash = new ColorHash({lightness: 0.5, saturation: 0.4});
    } else {
      console.error('Class \'ColorHash\' not defined.');
    }
  }

  /**
   *
   */
  setData(gpuData, cpuData) {
    const container = document.getElementById('timings');

    if (!container.matches(':hover')) {

      gpuData = JSON.parse(gpuData);
      cpuData = JSON.parse(cpuData);

      // Retrieve the end time values of the last root-level timing ranges.
      let maxGPUTime = gpuData[0][gpuData[0].length - 1][2];
      let maxCPUTime = cpuData[0][cpuData[0].length - 1][2];
      let maxTime    = Math.max(maxGPUTime, maxCPUTime) * 0.001;

      const item     = document.getElementById('fps-counter');
      item.innerHTML = `FPS: ${(1000.0 / maxTime).toFixed(2)} / ${(maxTime).toFixed(2)} ms`;

      this._drawGrid(maxTime);
      this._drawRanges("#gpu-ranges", gpuData, maxTime);
      this._drawRanges("#cpu-ranges", cpuData, maxTime);
    }
  }

  /**
   * Insert the actual html
   *
   */
  _drawRanges(selector, data, maxTime) {
    let container = document.querySelector(selector);
    CosmoScout.gui.clearHtml(container);

    let html = "";

    for (let i = 0; i < data.length; i++) {
      const level = data[i];

      if (level) {
        html += "<div class='level'>";
        for (let j = 0; j < level.length; j++) {
          let name     = level[j][0];
          let duration = (level[j][2] - level[j][1]) * 0.001;
          let start    = (level[j][1] * 0.001) / maxTime * 100;
          let end      = (level[j][2] * 0.001) / maxTime * 100;

          html += `<div class="range" data-tooltip="${name} (${
              duration.toFixed(
                  2)} ms)" style="--tooltip-offset:${(start + end) / 2}%; left: ${start}%; width: ${
              end - start}%; background-color: ${this._colorHash.hex(name)}"></div>`;
        }
        html += "</div>";
      }
    }

    const content     = document.createElement('template');
    content.innerHTML = html;
    container.appendChild(content.content);
  }

  _drawGrid(maxTime) {
    let grid = document.querySelector("#grid");
    CosmoScout.gui.clearHtml(grid);

    let interval = 1;
    let i        = 0;

    while (maxTime / interval > 50) {
      ++i

      let int = ~~(i / 3);
      let mod = i % 3;

      interval = (mod == 1 ? i * 2 : i * 2 + 1) * (int + 1);
    }

    let ticks       = Math.floor(maxTime / interval);
    const gridLines = document.createElement('template');

    for (let i = 0; i <= ticks; i++) {

      let margin = "";
      if (i == ticks) {
        let fullTickWidth = 100.0 / (ticks + 1);
        let remainder     = maxTime / interval - ticks;
        margin            = `style="margin-right: ${fullTickWidth * remainder}%"`;
      }

      if (i % 5 == 0) {
        gridLines.innerHTML +=
            `<div class="tick major" ${margin}><span>${i * interval} ms</span></div>`
      } else {
        gridLines.innerHTML += `<div class="tick minor" ${margin}></div>`;
      }
    }

    grid.appendChild(gridLines.content);
  }
}
