////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// The Int has a single output socket and a custom widget for entering a number. The
// custom widget is defined further below.
// The IntComponent serves as a kind of factory. Whenever a new node is created, the
// builder() method is called. It is required that the class is called <NAME>Component.
class CoverageViewerComponent extends Rete.Component {
  constructor() {
    // This name must match the Int::sName defined in Int.cpp.
    super("CoverageViewer");

    // This specifies the submenu from which this node can be created in the node editor.
    this.category = "Output";
  }

  // Called whenever a new node of this type needs to be constructed.
  builder(node) {

    let coverageInput = new Rete.Input('coverageIn', "Coverage", CosmoScout.socketTypes['Coverage']);
    node.addInput(coverageInput);

    let control = new CoverageViewerControl("CoverageViewer");
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
          return;
        }
        
        control.displayBounds((message["bounds"] ? message["bounds"] : {}));
        control.displayKeywords((message["keywords"] ? message["keywords"] : ""));
        control.displayAbstract((message["abstract"] ? message["abstract"] : ""));
        control.displayAttribution((message["attribution"] ? message["attribution"] : ""));
        control.displayTimeInterval((message["intervals"] ? message["intervals"] : []));
      };
    };

    return node;
  }
}

// This is the widget which is used for inserting the number.
class CoverageViewerControl extends Rete.Control {
  constructor(key) {
    super(key);

    this.id = crypto.randomUUID();

    // This HTML code will be used whenever a node is created with this widget.
    this.template = `
      <div class="container-fluid" style="width: 350px">
        <div class="row">
          <div class="col"><b>Bounds:</b></div>
        </div>
        <div style="width: fit-content;">
          <div class="row">
            <div class="col-auto">Longitude Min:</div>
            <div class="col-auto" id="${this.id}-minLong"></div> 
          </div>

          <div class="row">
            <div class="col-auto">Longitude Max:</div>
            <div class="col-auto" id="${this.id}-maxLong"></div> 
          </div>

          <div class="row">
            <div class="col-auto">Latitude Min:&nbsp;&nbsp;</div>
            <div class="col-auto" id="${this.id}-minLat"></div> 
          </div>

          <div class="row">
            <div class="col-auto">Latitude Max:&nbsp;&nbsp;</div>
            <div class="col-auto" id="${this.id}-maxLat"></div> 
          </div>
        </div>

        <br>
        <div class="row">
          <div class="col"><b>Keywords:</b></div>
        </div>
        <div class="row">
          <div class="col-auto" id="${this.id}-keywords"></div> 
        </div>

        <br>
        <div class="row">
          <div class="col"><b>Abstract:</b></div>
        </div>
        <div class="row">
          <div class="col-auto" id="${this.id}-abstract"></div> 
        </div>

        <br>
        <div class="row">
          <div class="col"><b>Attribution:</b></div>
        </div>
        <div class="row">
          <div class="col-auto" id="${this.id}-attribution"></div> 
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
  }

  /**
   * Displays the geographic coordinates.
   * @param {Object} bounds bounds object containing minLong, maxLong, minLat and maxLat 
   */
  displayBounds(bounds) {
    if (!bounds["minLong"]) {
      this.minLong.innerHTML = "";
      this.maxLong.innerHTML = "";
      this.minLat.innerHTML  = "";
      this.maxLat.innerHTML  = "";  
    } else {
      this.minLong.innerHTML = bounds["minLong"];
      this.maxLong.innerHTML = bounds["maxLong"];
      this.minLat.innerHTML  = bounds["minLat"];
      this.maxLat.innerHTML  = bounds["maxLat"];
    }
  }

  /**
   * Display the keywords of a coverage.
   * @param {String} keywords keywords in a comma separated list 
   */
  displayKeywords(keywords) {
    this.keywords.innerHTML = keywords;
  }

  /**
   * Displays the abstract of a coverage
   * @param {String} abstract abstract to display 
   */
  displayAbstract(abstract) {
    this.abstract.innerHTML = abstract;
  }

  /**
   * Displays the attribution of a coverage
   * @param {String} attribution attribution to display
   */
  displayAttribution(attribution) {
    this.attribution.innerHTML = attribution;
  }

  /**
   * Displays all available time intervals of a coverage
   * @param {Array<String>} timeIntervals time intervals to display
   */
  displayTimeInterval(timeIntervals) {
    this.timeInterval.innerHTML = "";
    for (const interval of timeIntervals) {
      this.timeInterval.innerHTML += interval + "\n";
    }
  }
}