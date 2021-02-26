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
   * Both arguments should be JSON strings containing an array for each nesting level. Each element
   * of these should contain an array of timing ranges. Each timing range is an array of three
   * elements: [<name>, <frame-relative-start>, <frame-relative-end>].
   */
  setData(gpuData, cpuData) {
    const container = document.getElementById('timings');

    // Only update the graph if it's not hovered.
    if (!container.matches(':hover')) {

      gpuData = JSON.parse(gpuData);
      cpuData = JSON.parse(cpuData);

      // Retrieve the end time values of the last root-level timing ranges. The maximum of these
      // determines the maximum x-value of the graph.
      let maxGPUTime = gpuData[0][gpuData[0].length - 1][2];
      let maxCPUTime = cpuData[0][cpuData[0].length - 1][2];
      let maxTime    = Math.max(maxGPUTime, maxCPUTime) * 0.001;

      // With this value we can update the FPS display.
      const item     = document.getElementById('fps-counter');
      item.innerHTML = `FPS: ${(1000.0 / maxTime).toFixed(2)} / ${(maxTime).toFixed(2)} ms`;

      // First we update the background grid.
      this._drawGrid(maxTime);

      // Then we draw the two graphs.
      this._drawRanges("#gpu-ranges", gpuData, maxTime);
      this._drawRanges("#cpu-ranges", cpuData, maxTime);
    }
  }

  /**
   * Draw the ranges of each level as small containers with a relative with and position.
   *
   * @param {string} selector The container into which the range bars are drawn.
   * @param {array}  data     The parsed JSON string passed to setData().
   * @param {number} maxTime  The maximum x-value of the graph.
   */
  _drawRanges(selector, data, maxTime) {

    // First clear the container completely.
    let container = document.querySelector(selector);
    CosmoScout.gui.clearHtml(container);

    // This string will contain all the HTML of the graph.
    let html = "";

    // Iterate through the nesting levels.
    for (let i = 0; i < data.length; i++) {
      const level = data[i];

      // If there are ranges for this level...
      if (level) {

        html += "<div class='level'>";

        // ... add one container for each.
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

    // Add the HTML to the document.
    const content     = document.createElement('template');
    content.innerHTML = html;
    container.appendChild(content.content);
  }

  /**
   * Draw a grid with major and minor ticks.
   *
   * @param {number} maxTime
   */

  _drawGrid(maxTime) {

    // First clear the container completely.
    let grid = document.querySelector("#grid");
    CosmoScout.gui.clearHtml(grid);

    // First we determine a suitable interval between the minor ticks. We first assume 1(ms) but
    // increase this if maxTime is quite large.
    let interval = 1;
    let i        = 0;

    // If the current interval would result in more than 50 ticks, we increase the interval.
    while (maxTime / interval > 50) {
      ++i

      // For linearly increasing i, this results in these intervals:
      // 1 2 5 10 20 50 100 200 500 1000 ....
      let int  = ~~(i / 3);
      let mod  = i % 3;
      interval = (mod === 1 ? i * 2 : i * 2 + 1) * (int + 1);
    }

    // Compute the number of required grid lines.
    let ticks       = Math.floor(maxTime / interval);
    const gridLines = document.createElement('template');

    for (let i = 0; i <= ticks; i++) {

      // The last tick line will require a little bit of margin on the right hand side as the
      // maxTime will not be a full multiple of interval.
      let margin = "";
      if (i === ticks) {
        let fullTickWidth = 100.0 / (ticks + 1);
        let remainder     = maxTime / interval - ticks;
        margin            = `style="margin-right: ${fullTickWidth * remainder}%"`;
      }

      // Every 5 ticks we add a major grid line.
      if (i % 5 === 0) {
        gridLines.innerHTML +=
            `<div class="tick major" ${margin}><span>${i * interval} ms</span></div>`
      } else {
        gridLines.innerHTML += `<div class="tick minor" ${margin}></div>`;
      }
    }

    grid.appendChild(gridLines.content);
  }
}
