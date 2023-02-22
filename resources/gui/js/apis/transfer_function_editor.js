/* global IApi, CosmoScout */

// This transfer function editor is heavily based on the tf-editor component
// (https://github.com/VolumeRC/tf-editor). The code was modified to be usable without using web
// components.
// Copyright(c) 2016 Vicomtech - IK4.All rights reserved.This code may only be used
// under the Apache License 2.0 found at http://volumerc.github.io/tf-editor/LICENSE.txt
// The complete set of attributions may be found at http://volumerc.github.io/tf-editor/NOTICE

/**
 * Class for managing a transfer function editor.
 */
class TransferFunctionEditor {
  /**
   * Constructs a transfer function editor on the given element.
   * Options can be passed in one object as the fourth parameter.
   * Available options:
   *  Width:           The width of the editor in pixels
   *  Height:          The height of the editor in pixels
   *  FitToData:       Whether the min and max x values should be determined by the data
   *  NumberTicks:     Number of ticks on x and y axis
   *  DefaultFunction: Will try to load this transfer function after initialization
   *
   * @param id {number} ID of the editor
   * @param element {HTMLElement} Root element of the editor
   * @param callback {(s: string) => void} Callback that should be called on changes in the editor
   * @param options {{width: number, height: number, fitToData: bool, numberTicks: number,
   *     defaultFunction: string}} Options for the editor
   */
  constructor(id, element, callback, {
    width           = 400,
    height          = 150,
    fitToData       = false,
    numberTicks     = 5,
    numberBins      = 100,
    defaultFunction = "",
    logScaleEnabled = false
  } = {}) {
    this.id       = id;
    this.element  = element;
    this.callback = callback;
    this.options  = {
      width: width,
      height: height,
      fitToData: fitToData,
      numberTicks: numberTicks,
      numberBins: numberBins,
      logScaleEnabled: logScaleEnabled
    }; //, defaultFunction: defaultFunction};
    this._data = [];

    this._initialized = false;
    this._createElements();
    this._initializeElements();
    this._drawChart();
    this._initialized = true;

    if (defaultFunction != "") {
      CosmoScout.callbacks.transferFunctionEditor.importTransferFunction(defaultFunction, this.id);
    }
  }

  /**
   * Sets the data for the visualization that is controlled by this editor.
   * This updates the values on the x axis if fitToData is set in the options.
   *
   * @param data {number[]} Array of scalar values
   */
  setData(data, resetSliderHandles = true, extents = undefined) {
    if (!Array.isArray(data)) {
      data = JSON.parse(data);
    }
    this._initialized = false;
    if (extents) {
      this._setExtents(extents);
    } else {
      this._setExtents(data);
    }
    this._updateScales(resetSliderHandles);
    this._data = this._bins(data).map((b) => { return {x0: b.x0, x1: b.x1, count: b.length}; });
    this._updateAxis();
    if (resetSliderHandles) {
      this._updateControlPoints(this._controlPoints);
    }
    this._initialized = true;
    this._redraw();
    this._redrawHistogram();
  }

  addData(data, resetSliderHandles = true) {
    if (!this._data || this._data.length <= 0) {
      this.setData(data, resetSliderHandles);
      return;
    }
    if (!Array.isArray(data)) {
      data = JSON.parse(data);
    }
    this._initialized = false;
    this._updateScales(resetSliderHandles);
    const newData = this._bins(data);
    for (let i = 0; i < newData.length; i++) {
      if (this._data[i]) {
        this._data[i].count += newData[i].length;
      } else {
        this._data[i] = {x0: newData[i].x0, x1: newData[i].x1, count: newData[i].length};
      }
    }
    this._updateAxis();
    if (resetSliderHandles) {
      this._updateControlPoints(this._controlPoints);
    }
    this._initialized = true;
    this._redraw();
    this._redrawHistogram();
  }

  clearData(resetSliderHandles = true) {
    this.setData([], resetSliderHandles);
  }

  _setExtents(data) {
    if (this.options.fitToData && data && data.length > 0) {
      this._dataExtent = d3.extent(data);
      if (this._dataExtent[0] == this._dataExtent[1]) {
        this._dataExtent[1] += 1;
      }
      return;
    }
    this._dataExtent = [0, 100];
  }

  _createXRangeSlider(range, resetHandles) {
    let start = range;
    if (this._xRangeSlider.noUiSlider !== undefined) {
      if (!resetHandles) {
        start = this._xRangeSlider.noUiSlider.get();
      }
      this._xRangeSlider.noUiSlider.destroy();
    }
    noUiSlider.create(this._xRangeSlider, {
      range: {"min": range[0], "max": range[1]},
      start: start,
      margin: (range[1] - range[0]) / 100
    });
    this._xRangeSlider.noUiSlider.on("update", (values, handle, unencoded) => {
      this._xScale.domain([unencoded[0], unencoded[1]]);
      if (this._initialized) {
        this._updateAxis();
        this._redraw();
        this._redrawHistogram();
      }
    });
  }

  _createElements() {
    // Range sliders
    this._xRangeSlider =
        this.element.querySelector("#transferFunctionEditor\\.xRangeSlider-" + this.id);
    this._yRangeSlider =
        this.element.querySelector("#transferFunctionEditor\\.yRangeSlider-" + this.id);

    // Axis scales
    this._xScale   = d3.scaleLinear();
    this._yScale   = d3.scaleLinear();
    this._binScale = d3.scaleLog();

    // Area for the opacity map representation
    this._area = d3.area();

    // Create histogram object
    this._bins = d3.histogram();

    // Keep track of control points interaction
    this._dragged    = null;
    this._selected   = null;
    this._last_color = 'green';

    this._controlPoints = [];

    // Custom margins
    this._margin = {top: 10, right: 20, bottom: 25, left: 40};

    // Access the svg dom element
    this._svg = d3.select(this.element)
        .select("#transferFunctionEditor\\.graph-" + this.id)
        .attr("width", this.options.width)
        .attr("height", this.options.height);
    this._width  = +this._svg.attr("width") - this._margin.left - this._margin.right;
    this._height = +this._svg.attr("height") - this._margin.top - this._margin.bottom;
  }

  _initializeElements() {
    this._createXRangeSlider([0, 100], true);
    noUiSlider.create(
        this._yRangeSlider, {range: {"min": 0, "max": 1}, start: [0, 1], margin: 0.01});
    this._yRangeSlider.noUiSlider.on("update", (values, handle, unencoded) => {
      this._yScale.domain([unencoded[0], unencoded[1]]);
      if (this._initialized) {
        this._updateAxis();
        this._redraw();
      }
    });

    const pickerDiv =
        this.element.querySelector("#transferFunctionEditor\\.colorPicker-" + this.id);
    pickerDiv.picker = new CP(pickerDiv);
    pickerDiv.picker.self.classList.add('no-alpha');
    pickerDiv.picker.on('change', (r, g, b, a) => {
      const color                = CP.HEX([r, g, b, 1]);
      pickerDiv.style.background = color;
      pickerDiv.value            = color;
    });

    pickerDiv.oninput = (e) => {
      const color = CP.HEX(e.target.value);
      pickerDiv.picker.set(color[0], color[1], color[2], 1);
      pickerDiv.style.background = CP.HEX([color[0], color[1], color[2], 1]);
    };

    this._setExtents(this._data);

    this._xScale.rangeRound([0, this._width]).domain(this._dataExtent);
    this._yScale.domain([0, 1]).range([this._height, 0]);
    this._binScale.domain([1, 10]).range([this._height, 0]).base(2).clamp([0, this._height]);
    this._bins.domain(this._dataExtent).thresholds(this.options.numberBins);

    if (this._controlPoints.length == 0) {
      this._controlPoints.push(
          {'x': this._dataExtent[0], 'opacity': 0, 'color': '#0000FF', 'locked': true});
      this._controlPoints.push(
          {'x': this._dataExtent[1], 'opacity': 1, 'color': '#FF0000', 'locked': true});
    }
    this._selected = this._controlPoints[1];
    this._area.x(d => { return this._xScale(d.x); })
        .y0(d => { return this._yScale(d.opacity); })
        .y1(this._height);

    // Access the color selector
    this._colorPicker =
        $(this.element).find("#transferFunctionEditor\\.colorPicker-" + this.id).get(0);
    this._colorPicker.picker.on("change", () => {
      this._selected.color = this._colorPicker.value;
      this._redraw();
    });
    // Export button listener
    $(this.element).find("#transferFunctionEditor\\.export-" + this.id).on("click", () => {
      CosmoScout.callbacks.transferFunctionEditor.exportTransferFunction(
          $(this.element).find("#transferFunctionEditor\\.exportLocation-" + this.id).val(),
          this.getJsonString());
    });
    // Import button listener
    $(this.element).find("#transferFunctionEditor\\.import-" + this.id).on("click", () => {
      CosmoScout.callbacks.transferFunctionEditor.importTransferFunction(
          $(this.element).find("#transferFunctionEditor\\.importSelect-" + this.id).val(), this.id);
    });

    // Lock button listener
    $(this.element).find("#transferFunctionEditor\\.colorLock-" + this.id).on("click", () => {
      if (this._controlPoints.some(point => point.locked && point !== this._selected)) {
        this._selected.locked = !this._selected.locked;
        this._redraw();
        this._updateLockButtonState();
      }
    });

    // Log scale checkbox listener
    $(this.element).find("#transferFunctionEditor\\.logScale-" + this.id).on("click", (event) => {
      this.options.logScaleEnabled = event.currentTarget.checked;
      CosmoScout.callbacks.transferFunctionEditor.importTransferFunction(
          $(this.element).find("#transferFunctionEditor\\.importSelect-" + this.id).val(), this.id);
    });
  }

  _updateLockButtonState() {
    const colorLockButton =
        $(this.element).find("#transferFunctionEditor\\.colorLock-" + this.id + " i");
    if (this._selected.locked) {
      $(colorLockButton).html("lock");
    } else {
      $(colorLockButton).html("lock_open");
    }
  }

  // Perform the drawing
  _drawChart() {
    const g =
        this._svg.append("g")
            .attr("transform", "translate(" + this._margin.left + "," + this._margin.top + ")")
            .on("mouseleave", () => { this._mouseup(); });

    this._redrawHistogram();

    // Gradient definitions
    g.append("defs")
        .append("linearGradient")
        .attr("id", "transferFunctionEditor.gradient-" + this.id)
        .attr("gradientUnits", "objectBoundingBox")
        .attr("spreadMethod", "pad")
        .attr("x1", "0%")
        .attr("y1", "0%")
        .attr("x2", "100%")
        .attr("y2", "0%");

    // Draw graph
    const graph = g.append("svg")
        .attr("width", this._width)
        .attr("height", this._height)
        .attr("overflow", "hidden");

    graph.append("path")
        .datum(this._controlPoints)
        .attr("class", "line")
        .attr("fill", "url(#transferFunctionEditor.gradient-" + this.id + ")");
    graph.append("g").attr("class", "histogram-group");
    graph.append("path").datum(this._controlPoints).attr("class", "line").attr("fill", "none");

    // Draw axis
    const xTicks              = this._xScale.ticks(this.options.numberTicks);
    xTicks[xTicks.length - 1] = this._xScale.domain()[1];
    this._xAxis               = g.append("g")
        .attr("class", "axis axis--x")
        .attr("transform", "translate(0," + this._height + ")")
        .call(d3.axisBottom(this._xScale).tickValues(xTicks));

    this._yAxis = g.append("g")
        .attr("class", "axis axis--y")
        .attr("transform", "translate(0, 0)")
        .call(d3.axisLeft(this._yScale).ticks(this.options.numberTicks));

    // Mouse interaction handler
    g.append("rect")
        .attr("y", -10)
        .attr("x", -10)
        .attr("width", this._width + 20)
        .attr("height", this._height + 20)
        .style("opacity", 0)
        .on("mousedown", () => { this._mousedown(); })
        .on("mouseup", () => { this._mouseup(); })
        .on("mousemove", () => { this._mousemove(); });
  }

  // update scales with new data input
  _updateScales(resetSliderHandles) {
    this._xScale.domain(this._dataExtent);
    this._createXRangeSlider(this._dataExtent, resetSliderHandles);
    this._bins.domain(this._dataExtent);
  }

  // update the axis with the new data input
  _updateAxis() {
    const svg                 = d3.select("svg").select("g");
    const xTicks              = this._xScale.ticks(this.options.numberTicks);
    xTicks[xTicks.length - 1] = this._xScale.domain()[1];
    this._xAxis.call(d3.axisBottom(this._xScale).tickValues(xTicks));
    const yTicks              = this._yScale.ticks(this.options.numberTicks);
    yTicks[yTicks.length - 1] = this._yScale.domain()[1];
    this._yAxis.call(d3.axisLeft(this._yScale).tickValues(yTicks));
  }

  _updateControlPoints(controlPoints) {
    const updateScale = d3.scaleLinear()
        .domain(d3.extent(controlPoints, point => point.x))
        .range(this._xScale.domain());
    controlPoints.forEach(point => point.x = updateScale(point.x));
    this._controlPoints = controlPoints;
  }

  // Update the chart content
  _redraw() {
    if (!this._controlPoints.some(point => point.locked)) {
      this._selected.locked = !this._selected.locked;
      this._redraw();
      this._updateLockButtonState();
    }
    this._controlPoints.forEach((point, index) => {
      if (index == 0) {
        point.x = this._dataExtent[0];
      } else if (index == this._controlPoints.length - 1) {
        point.x = this._dataExtent[1];
      }

      if (!point.locked) {
        const right = this._controlPoints.slice(index, this._controlPoints.length)
            .find(point => point.locked);
        const left = this._controlPoints.slice(0, index).reverse().find(point => point.locked);
        if (left && right) {
          point.color =
              d3.interpolateRgb(left.color, right.color)((point.x - left.x) / (right.x - left.x));
        } else if (left) {
          point.color = left.color;
        } else if (right) {
          point.color = right.color;
        }
      }
    });

    const svg =
        d3.select(this.element).select("#transferFunctionEditor\\.graph-" + this.id).select("g");
    svg.selectAll("path.line").datum(this._controlPoints).attr("d", this._area);

    // Add circle to connect and interact with the control points
    const circle = svg.selectAll("circle").data(this._controlPoints);

    circle.style("visibility", (d) => {
      const x = this._xScale(d.x);
      const y = this._yScale(d.opacity);
      if (x < 0 || x > this._width || y < 0 || y > this._height) {
        return "hidden";
      } else {
        return "visible";
      }
    });

    circle.enter()
        .append("circle")
        .attr("cx", (d) => { return this._xScale(d.x); })
        .attr("cy", (d) => { return this._yScale(d.opacity); })
        .style("fill", (d) => { return d.color; })
        .attr("r", (d) => { return d.locked ? 6.0 : 4.0; })
        .on("mousedown",
            (d) => {
              this._selected = this._dragged = d;
              this._last_color               = d.color;
              this._redraw();
              this._updateLockButtonState();
            })
        .on("mouseup", () => { this._mouseup(); })
        .on("mousemove", () => { this._mousemove(); })
        .on("contextmenu", (d, i) => {
          // react on right-clicking
          d3.event.preventDefault();
          d.color  = this.colorPicker.value;
          d.locked = true;
          this._redraw();
          this._updateLockButtonState();
        });

    circle.classed("selected", (d) => { return d === this._selected; })
        .style("fill", (d) => { return d.color; })
        .attr("cx", (d) => { return this._xScale(d.x); })
        .attr("cy", (d) => { return this._yScale(d.opacity); })
        .attr("r", (d) => { return d.locked ? 6.0 : 4.0; });

    circle.exit().remove();

    // Create a linear gradient definition of the control points
    const gradient = svg.select("linearGradient").selectAll("stop").data(this._controlPoints);

    gradient.enter()
        .append("stop")
        .attr("stop-color", (d) => { return d.color; })
        .attr("stop-opacity", (d) => { return d.opacity; })
        .attr("offset", (d) => {
          const l =
              (this._controlPoints[this._controlPoints.length - 1].x - this._controlPoints[0].x);
          return "" + ((d.x - this._controlPoints[0].x) / l * 100) + "%";
        });

    gradient.attr("stop-color", (d) => {
      return d.color;
    }).attr("stop-opacity", (d) => { return d.opacity; }).attr("offset", (d) => {
      const l = (this._controlPoints[this._controlPoints.length - 1].x - this._controlPoints[0].x);
      return "" + ((d.x - this._controlPoints[0].x) / l * 100) + "%";
    });

    gradient.exit().remove();

    if (this.callback !== undefined) {
      this.callback(this.getJsonString());
    }

    if (d3.event) {
      d3.event.preventDefault();
      d3.event.stopPropagation();
    }
  }

  _redrawHistogram() {
    this._svg.select("g").select(".histogram-group").selectAll(".bar").remove();
    if (this._data && this._data.length > 0) {
      this._binScale.domain([0.1, d3.max(this._data, (d) => { return d.count; })]);
      const bar =
          this._svg.select("g").select(".histogram-group").selectAll(".bar").data(this._data);
      const barEnter = bar.enter().append("g").attr("class", "bar").attr("transform", (d) => {
        return "translate(" + this._xScale(d.x0) + "," + this._binScale(d.count) + ")";
      });

      barEnter.append("rect")
          .attr("x", 1)
          .style("opacity", 0.5)
          .attr("width", (d) => { return this._xScale(d.x1) - this._xScale(d.x0); })
          .attr("height", (d) => { return this._height - this._binScale(d.count); });

      this._svg.select("g").select(".histogram-group").selectAll(".bar").lower();

      bar.exit().remove();
    }
  }

  /////// User interaction related event callbacks ////////

  _mousedown() {
    const pos   = d3.mouse(this._svg.node());
    const point = {
      "x": this._xScale.invert(Math.max(0, Math.min(pos[0] - this._margin.left, this._width))),
      "opacity": this._yScale.invert(Math.max(0, Math.min(pos[1] - this._margin.top, this._height)))
    };
    this._selected = this._dragged = point;
    const bisect                   = d3.bisector((a, b) => { return a.x - b.x; }).left;
    const indexPos                 = bisect(this._controlPoints, point);
    this._controlPoints.splice(indexPos, 0, point);
    this._redraw();
    this._updateLockButtonState();
  }

  _mousemove() {
    if (!this._dragged)
      return;

    function equalPoint(a, index, array) {
      return a.x == this.x && a.opacity == this.opacity && a.color == this.color;
    };
    const index = this._controlPoints.findIndex(equalPoint, this._selected);
    if (index == -1)
      return;
    const m        = d3.mouse(this._svg.node());
    this._selected = this._dragged = this._controlPoints[index];
    if (index != 0 && index != this._controlPoints.length - 1) {
      this._dragged.x =
          this._xScale.invert(Math.max(0, Math.min(this._width, m[0] - this._margin.left)));
    }
    this._dragged.opacity =
        this._yScale.invert(Math.max(0, Math.min(this._height, m[1] - this._margin.top)));
    const bisect        = d3.bisector((a, b) => { return a.x - b.x; }).left;
    const bisect2       = d3.bisector((a, b) => { return a.x - b.x; }).right;
    const virtualIndex  = bisect(this._controlPoints, this._dragged);
    const virtualIndex2 = bisect2(this._controlPoints, this._dragged);
    if (virtualIndex < index) {
      this._controlPoints.splice(virtualIndex, 1);
    } else if (virtualIndex > index) {
      this._controlPoints.splice(index + 1, 1);
    } else if (virtualIndex2 - index >= 2) {
      this._controlPoints.splice(index + 1, 1);
    }
    this._redraw();
  }

  _mouseup() {
    if (!this._dragged)
      return;
    this._dragged = null;
  }

  /**
   * @return {string} The current transfer function as a json string
   */
  getJsonString() {
    // Utility functions for parsing colors
    function colorHexToComponents(hexString) {
      const red   = parseInt(hexString.substring(1, 3), 16) / 255.0;
      const green = parseInt(hexString.substring(3, 5), 16) / 255.0;
      const blue  = parseInt(hexString.substring(5, 7), 16) / 255.0;
      return [red, green, blue];
    }

    function colorRgbToComponents(rgbString) {
      return rgbString.substring(4, rgbString.length - 1).split(",").map(s => s.trim() / 255.0);
    }

    function colorToComponents(colorString) {
      if (colorString.startsWith("#")) {
        return colorHexToComponents(colorString);
      } else if (colorString.startsWith("rgb")) {
        return colorRgbToComponents(colorString);
      }
    }

    const exportObject = {};
    exportObject.RGB   = {};
    exportObject.Alpha = {};

    const min   = this._dataExtent[0];
    const max   = this._dataExtent[1];
    const range = max - min;
    this._controlPoints.forEach((controlPoint) => {
      const position = controlPoint.x;
      const opacity  = controlPoint.opacity;
      if (controlPoint.locked) {
        exportObject.RGB[(position - min) / range] = colorToComponents(controlPoint.color);
      }
      exportObject.Alpha[(position - min) / range] = opacity;
    });

    return JSON.stringify(exportObject);
  }

  /**
   * Loads a transfer function into this editor.
   *
   * @param jsonTransferFunction {string} json string describing the transfer function
   */
  loadTransferFunction(jsonTransferFunction) {
    let   transferFunction = JSON.parse(jsonTransferFunction);
    const points           = [];
    if (Array.isArray(transferFunction)) {
      // Paraview transfer function format
      transferFunction = transferFunction[0];
      let index        = 0;
      for (let i = 0; i < transferFunction.RGBPoints.length;) {
        points[index]          = {};
        points[index].position = transferFunction.RGBPoints[i++];
        points[index].r        = transferFunction.RGBPoints[i++];
        points[index].g        = transferFunction.RGBPoints[i++];
        points[index].b        = transferFunction.RGBPoints[i++];
        index++;
      }
      if (transferFunction.Points != null) {
        for (let i = 0; i < transferFunction.Points.length;) {
          if (!points.some(
              point => Number(point.position) === Number(transferFunction.Points[i]))) {
            points[index]          = {};
            points[index].position = transferFunction.Points[i++];
            points[index].a        = transferFunction.Points[i++];
            i += 2;
            index++;
          } else {
            points.find(point => Number(point.position) === Number(transferFunction.Points[i])).a =
                transferFunction.Points[++i];
            i += 3;
          }
        }
      } else {
        points[0].a                 = 0;
        points[points.length - 1].a = 1;
      }
    } else {
      // Cosmoscout transfer function format
      let index = 0;
      Object.keys(transferFunction.RGB).forEach((position) => {
        points[index]          = {};
        points[index].position = position;
        points[index].r        = transferFunction.RGB[position][0];
        points[index].g        = transferFunction.RGB[position][1];
        points[index].b        = transferFunction.RGB[position][2];
        ++index;
      });
      Object.keys(transferFunction.Alpha).forEach((position) => {
        if (!points.some(point => Number(point.position) === Number(position))) {
          points[index]          = {};
          points[index].position = position;
          points[index].a        = transferFunction.Alpha[position];
          ++index;
        } else {
          points.find(point => Number(point.position) === Number(position)).a =
              transferFunction.Alpha[position];
        }
      });
    }

    points.sort((a, b) => { return a.position - b.position; });

    if (Number(Number(this._xRangeSlider.noUiSlider.get()[0]).toPrecision(4)) >
        Number(this._dataExtent[0].toPrecision(4))) {
      points.unshift(points[0]);
    }
    if (Number(Number(this._xRangeSlider.noUiSlider.get()[1]).toPrecision(4)) <
        Number(this._dataExtent[1].toPrecision(4))) {
      points.push(points[points.length - 1]);
    }

    const min = points[0].position;
    const max = points[points.length - 1].position;
    points.forEach((point, index) => {
      if (typeof point.a === 'undefined') {
        var right = points.slice(index, points.length).find(point => point.hasOwnProperty("a"));
        var left  = points.slice(0, index).reverse().find(point => point.hasOwnProperty("a"));
        if (left && right) {
          point.a = d3.interpolateNumber(left.a, right.a)(
              (point.position - left.position) / (right.position - left.position));
        } else if (left) {
          point.a = left.a;
        } else if (right) {
          point.a = right.a;
        }
      }
    });

    const min_cP = this._controlPoints[0].x;
    const max_cP = this._controlPoints[this._controlPoints.length - 1].x;

    this._updateControlPoints(points.map((point) => {
      var local_value = (max_cP - min_cP) * point.position + min_cP;
      var minLog      = isFinite(Math.log(min_cP)) ? Math.log(min_cP) : 0;
      var maxLog      = isFinite(Math.log(max_cP)) ? Math.log(max_cP) : 0;
      return {
        x: (this.options.logScaleEnabled)
            ? Math.pow(
                10, (local_value - min_cP) / (max_cP - min_cP) * (maxLog - minLog) + minLog)
            : (point.position - min) * (255.0 / (max - min)),
        opacity: point.a,
        color:
            (typeof point.r !== 'undefined')
                ? "#" + (0x1000000 + (point.b * 255 | (point.g * 255 << 8) | (point.r * 255 << 16)))
                .toString(16)
                .slice(1)
                : "#000000",
        locked: (typeof point.r !== 'undefined')
      };
    }));
    this._redraw();
  }

  /**
   * Updates the list of available files shown next to the import button.
   *
   * @param availableFiles {string[]} List of filenames
   */
  setAvailableTransferFunctions(availableFiles) {
    let options = "";
    availableFiles.forEach((file) => { options += `<option>${file}</option>`; });
    const importSelect = $(this.element).find("#transferFunctionEditor\\.importSelect-" + this.id);
    importSelect.html(options);
    importSelect.selectpicker();
    importSelect.selectpicker("refresh");
  }
}

/**
 * Transfer function editor API
 */
class TransferFunctionEditorApi extends IApi {
  /**
   * @inheritDoc
   */
  name = "transferFunctionEditor";

  /**
   * id suffix of the next created transfer function editor
   */
  _nextId = 0;

  /**
   * List of created transfer function editors
   * @type {TransferFunctionEditor[]}
   */
  _editors = [];

  /**
   * List of available transfer function files
   * @type {string[]}
   */
  _availableFiles = [];

  /**
   * @inheritDoc
   */
  init() {
    const templateHtml        = `
        <div>
          <div class="row">
            <div class="col-7 offset-5">
              <label class="checklabel">
                <input type="checkbox" data-initialized="true" id="transferFunctionEditor.logScale-%ID%" />
                <i class="material-icons"></i>
                <span>Enable log scale</span>
              </label>
            </div>
          </div>
          <div class="row">
            <div class="col-12">
              <svg id="transferFunctionEditor.graph-%ID%"></svg>
            </div>
          </div>
          <div class="row">
            <div class="col-5">
              X-Range
            </div>
            <div class="col-7">
              <div id="transferFunctionEditor.xRangeSlider-%ID%"></div>
            </div>
          </div>
          <div class="row">
            <div class="col-5">
              Y-Range
            </div>
            <div class="col-7">
              <div id="transferFunctionEditor.yRangeSlider-%ID%"></div>
            </div>
          </div>
          <div class="row">
            <div class="col-2">
              <a id="transferFunctionEditor.colorLock-%ID%" class="btn glass block" title="Lock Color">
                <i class="material-icons">lock</i>
              </a>
            </div>
            <div class="col-3">
              <input type="text" class="form-control color-input" id="transferFunctionEditor.colorPicker-%ID%" style="color: black;" value="#FF0000">
            </div>
          </div>
          <div class="row">
            <div class="col-5">
              <button id="transferFunctionEditor.export-%ID%" class="waves-effect waves-light block btn glass text">Export</button>
            </div>
            <div class="col-7">
              <input type="text" id="transferFunctionEditor.exportLocation-%ID%" placeholder="Filename" class="text-input text">
            </div>
          </div>
          <div class="row">
            <div class="col-5">
              <button id="transferFunctionEditor.import-%ID%" class="waves-effect waves-light block btn glass text">Import</button>
            </div>
            <div class="col-7" id="transferFunctionEditor.importSelectParent-%ID%">
              <select id="transferFunctionEditor.importSelect-%ID%">
                <option value="-1">none</option>
              </select>
            </div>
          </div>
        </div>
      `;
    CosmoScout.gui.addTemplate("transferFunctionEditor", templateHtml);
  }

  /**
   * Create a new transfer function editor.
   * See constructor of TransferFunctionEditor for a description of the available options.
   *
   * @param container {HTMLELement} HTML element in which the editor should be placed
   * @param callback {(s: string) => void} Callback that should be called on changes in the editor
   * @param options {{width: number, height: number, fitToData: bool, numberTicks: number,
   *     defaultFunction: string}} Options for the editor
   * @return {TransferFunctionEditor}
   */
  create(container, callback, options) {
    const editor = CosmoScout.gui.loadTemplateContent("transferFunctionEditor");
    if (editor === false) {
      console.warn('"#transferFunctionEditor-template" could not be loaded!');
      return;
    }

    container.innerHTML = editor.innerHTML.replace(/%ID%/g, this._nextId);

    const transferFunctionEditor =
        new TransferFunctionEditor(this._nextId, container, callback, options);

    transferFunctionEditor.setAvailableTransferFunctions(this._availableFiles);
    this._editors.push(transferFunctionEditor);
    this._nextId++;
    if (CosmoScout.callbacks.transferFunctionEditor !== undefined) {
      CosmoScout.callbacks.transferFunctionEditor.getAvailableTransferFunctions();
    }
    return transferFunctionEditor;
  }

  /**
   * Sets the list of available transfer function files for all created editors.
   *
   * @param availableFilesJson {string[]} List of transfer function filenames
   */
  setAvailableTransferFunctions(availableFilesJson) {
    this._availableFiles = JSON.parse(availableFilesJson);
    this._editors.forEach(
        (editor) => { editor.setAvailableTransferFunctions(this._availableFiles); });
  }

  /**
   * Adds one transfer function filename to the list of available files.
   * Afterwards updates the available functions for all editors.
   *
   * @param filename {string} Name of the file to be added
   */
  addAvailableTransferFunction(filename) {
    if (!this._availableFiles.includes(filename)) {
      let files = this._availableFiles;
      files.push(filename);
      files.sort();
      this.setAvailableTransferFunctions(JSON.stringify(files));
    }
  }

  /**
   * Loads a transfer function to one editor.
   *
   * @param jsonTransferFunction {string} json string describing the transfer function
   * @param editorId {number} ID of the editor for which the function should be loaded
   */
  loadTransferFunction(jsonTransferFunction, editorId) {
    this._editors.find(e => e.id == editorId).loadTransferFunction(jsonTransferFunction);
  }
}
