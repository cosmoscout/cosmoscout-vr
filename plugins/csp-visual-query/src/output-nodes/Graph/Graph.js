////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// The Int has a single output socket and a custom widget for entering a number. The
// custom widget is defined further below.
// The IntComponent serves as a kind of factory. Whenever a new node is created, the
// builder() method is called. It is required that the class is called <NAME>Component.
class GraphComponent extends Rete.Component {
  constructor() {
    // This name must match the Int::sName defined in Int.cpp.
    super("Graph");

    // This specifies the submenu from which this node can be created in the node editor.
    this.category = "Output";
  }

  // Called whenever a new node of this type needs to be constructed.
  builder(node) {

    let coverageInput = new Rete.Input('dataIn', "Data", CosmoScout.socketTypes['Image1D']);
    node.addInput(coverageInput);

    let control = new GraphControl("Graph");
    node.addControl(control);

    // Once the HTML element for this node has been created, the node.onInit() method will be
    // called. This is used here to initialize the widget.
    node.onInit = (nodeDiv) => {
      control.init(nodeDiv, node.data);
      node.onMessageFromCPP = (message) => { control.setData(message.data); };
    };

    return node;
  }
}

// This is the widget which is used for inserting the number.
class GraphControl extends Rete.Control {
  constructor(key) {
    super(key);

    this.id = crypto.randomUUID();

    // This HTML code will be used whenever a node is created with this widget.
    this.template = `
      <div style="width: 350px; height: 250px" id="${this.id}">
        <svg width="350" height="250"></svg>
      </div>
        `;
  }

  // This is called by the node.onInit() above once the HTML element for the node has been
  // created.
  init(nodeDiv, data) {

    // Get our display elements
    this.div = nodeDiv.querySelector('[id="' + this.id + '"]');
    this.svg = d3.select(this.div).select("svg");

    // Initialize the graph elements
    this.margin = {top: 20, right: 20, bottom: 30, left: 40};
    this.width  = 350 - this.margin.left - this.margin.right;
    this.height = 250 - this.margin.top - this.margin.bottom;

    this.graph =
        this.svg.append("g").attr("transform", `translate(${this.margin.left},${this.margin.top})`);

    // Create scales
    this.xScale = d3.scaleLinear().range([0, this.width]);
    this.yScale = d3.scaleLinear().range([this.height, 0]);

    // Create axes
    this.xAxis = this.graph.append("g")
                     .attr("class", "axis")
                     .attr("transform", `translate(0,${this.height})`);
    this.yAxis = this.graph.append("g").attr("class", "axis");

    // Create line generator
    this.line = d3.line().x((d, i) => this.xScale(i)).y((d) => this.yScale(d));

    // Append path for the line
    this.path = this.graph.append("path")
                    .attr("fill", "none")
                    .attr("stroke", "steelblue")
                    .attr("stroke-width", 2);
  }

  /**
   * Displays the geographic coordinates.
   * @param {Object} bounds bounds object containing minLong, maxLong, minLat and maxLat
   */
  setData(data) {
    console.log("GraphControl::setData", data);

    // Update scales
    this.xScale.domain([0, data.length - 1]);
    this.yScale.domain([d3.min(data), d3.max(data)]);

    // Update axes
    this.xAxis.call(d3.axisBottom(this.xScale));
    this.yAxis.call(d3.axisLeft(this.yScale));

    // Update line path
    this.path.datum(data).attr("d", this.line);
  }
}