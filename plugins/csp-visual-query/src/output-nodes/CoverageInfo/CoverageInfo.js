////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// The Int has a single output socket and a custom widget for entering a number. The
// custom widget is defined further below.
// The IntComponent serves as a kind of factory. Whenever a new node is created, the
// builder() method is called. It is required that the class is called <NAME>Component.
class CoverageInfoComponent extends Rete.Component {
  constructor() {
    // This name must match the Int::sName defined in Int.cpp.
    super("CoverageInfo");

    // This specifies the submenu from which this node can be created in the node editor.
    this.category = "Output";
  }

  // Called whenever a new node of this type needs to be constructed.
  builder(node) {

    let coverageInput =
        new Rete.Input('coverageIn', "Coverage", CosmoScout.socketTypes['Coverage']);
    node.addInput(coverageInput);

    let control = new CoverageInfoControl("CoverageInfo");
    node.addControl(control);

    // Once the HTML element for this node has been created, the node.onInit() method will be
    // called. This is used here to initialize the widget.
    node.onInit = (nodeDiv) => {
      control.init(nodeDiv, node.data);

      node.onMessageFromCPP = (message) => {
        if (!message) {
          control.displayBounds({});
          control.displayAbstract("");
          control.displayAttribution("");
          control.displayKeywords("");
          control.displayTimeInterval([]);
          control.displaySize({});
          control.displayLayers("");
          return;
        }

        control.displayBounds((message["bounds"] ? message["bounds"] : {}));
        control.displayKeywords((message["keywords"] ? message["keywords"] : ""));
        control.displayAbstract((message["abstract"] ? message["abstract"] : ""));
        control.displayAttribution((message["attribution"] ? message["attribution"] : ""));
        control.displayTimeInterval((message["intervals"] ? message["intervals"] : []));
        control.displaySize((message["size"] ? message["size"] : {}));
        control.displayLayers((message["layers"] ? message["layers"] : ""));
      };
    };

    return node;
  }
}

// This is the widget which is used for inserting the number.
class CoverageInfoControl extends Rete.Control {
  constructor(key) {
    super(key);

    this.id = crypto.randomUUID();

    // This HTML code will be used whenever a node is created with this widget.
    this.template = `
      <div class="container-fluid" style="width: 330px">
        <div class="row">
          <div class="col"><b>Bounds:</b></div>
        </div>
        <div class="row">
          <div class="col">Longitude Min:</div>
          <div class="col" id="${this.id}-minLong"></div> 
        </div>
        <div class="row">
          <div class="col">Longitude Max:</div>
          <div class="col" id="${this.id}-maxLong"></div> 
        </div>
        <div class="row">
          <div class="col">Latitude Min:&nbsp;&nbsp;</div>
          <div class="col" id="${this.id}-minLat"></div> 
        </div>
        <div class="row">
          <div class="col">Latitude Max:&nbsp;&nbsp;</div>
          <div class="col" id="${this.id}-maxLat"></div> 
        </div>

        <br>
        <div class="row">
          <div class="col"><b>Size:</b></div>
          <div class="col"><span id="${this.id}-width"></span> x <span id="${
        this.id}-height"></span></div>
        </div>

        <br>
        <div class="row">
          <div class="col">Layer Count:</div>
          <div class="col" id="${this.id}-layers"></div> 
        </div>

        <br>
        <div class="row">
          <div class="col"><b>Keywords:</b></div>
        </div>
        <div class="row">
          <div class="col" id="${this.id}-keywords"></div> 
        </div>

        <br>
        <div class="row">
          <div class="col"><b>Abstract:</b></div>
        </div>
        <div class="row">
          <div class="col" id="${this.id}-abstract"></div> 
        </div>

        <br>
        <div class="row">
          <div class="col"><b>Attribution:</b></div>
        </div>
        <div class="row">
          <div class="col" id="${this.id}-attribution"></div> 
        </div>

        <br>
        <div class="row">
          <div class="col"><b>Time Intervals:</b></div>
        </div>
        <div class="row">
          <div class="col-auto" id="${this.id}-timeInterval"></div> 
        </div>
      </div>
        `;
  }

  // This is called by the node.onInit() above once the HTML element for the node has been
  // created.
  init(nodeDiv, data) {

    // Get our display elements
    this.minLong      = nodeDiv.querySelector('[id="' + this.id + '-minLong"]');
    this.maxLong      = nodeDiv.querySelector('[id="' + this.id + '-maxLong"]');
    this.minLat       = nodeDiv.querySelector('[id="' + this.id + '-minLat"]');
    this.maxLat       = nodeDiv.querySelector('[id="' + this.id + '-maxLat"]');
    this.keywords     = nodeDiv.querySelector('[id="' + this.id + '-keywords"]');
    this.abstract     = nodeDiv.querySelector('[id="' + this.id + '-abstract"]');
    this.attribution  = nodeDiv.querySelector('[id="' + this.id + '-attribution"]');
    this.timeInterval = nodeDiv.querySelector('[id="' + this.id + '-timeInterval"]');
    this.width        = nodeDiv.querySelector('[id="' + this.id + '-width"]');
    this.height       = nodeDiv.querySelector('[id="' + this.id + '-height"]');
    this.layers       = nodeDiv.querySelector('[id="' + this.id + '-layers"]');
  }

  /**
   * Displays the geographic coordinates.
   * @param {Object} bounds bounds object containing minLong, maxLong, minLat and maxLat
   */
  displayBounds(bounds) {
    if (!bounds["minLong"]) {
      this.minLong.innerText = "";
      this.maxLong.innerText = "";
      this.minLat.innerText  = "";
      this.maxLat.innerText  = "";
    } else {
      this.minLong.innerText = bounds["minLong"];
      this.maxLong.innerText = bounds["maxLong"];
      this.minLat.innerText  = bounds["minLat"];
      this.maxLat.innerText  = bounds["maxLat"];
    }
  }

  /**
   * Displays the size of the coverage.
   * @param {Object} size size object containing width and height
   */
  displaySize(size) {
    this.width.innerText  = size.width;
    this.height.innerText = size.height;
  }

  /**
   * Displays the number of layers of a coverage.
   */
  displayLayers(layers) {
    this.layers.innerText = layers;
  }

  /**
   * Display the keywords of a coverage.
   * @param {String} keywords keywords in a comma separated list
   */
  displayKeywords(keywords) {
    this.keywords.innerText = keywords;
  }

  /**
   * Displays the abstract of a coverage
   * @param {String} abstract abstract to display
   */
  displayAbstract(abstract) {
    this.abstract.innerText = abstract;
  }

  /**
   * Displays the attribution of a coverage
   * @param {String} attribution attribution to display
   */
  displayAttribution(attribution) {
    this.attribution.innerText = attribution;
  }

  /**
   * Displays all available time intervals of a coverage
   * @param {Array<String>} timeIntervals time intervals to display
   */
  displayTimeInterval(timeIntervals) {
    this.timeInterval.innerText = "";
    for (const interval of timeIntervals) {
      this.timeInterval.innerText += interval + "\n";
    }
  }
}