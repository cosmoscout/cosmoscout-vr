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
    // called. This is used here to initialize the input widget. The node.data object may
    // contain a number as returned by Int::getData() which - if present - should be
    // preselected.
    node.onInit = (nodeDiv) => { 
      
      control.init(nodeDiv, node.data); 
      
      node.onMessageFromCPP = (message) => {
        if (message["bounds"]) {
          control.updateValues(message["bounds"]["minLong"], message["bounds"]["maxLong"], 
            message["bounds"]["minLat"], message["bounds"]["maxLat"]);
        }
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
      <div class="container-fluid" style="width: 250px">
        <div class="row">
          <div class="col">Bounds:</div>
        </div>
        <hr>
        <div class="row">
          <div class="col-md-7">Longitude Min:</div>
          <div class="col-md-auto" id="${this.id}-minLong"></div> 
        </div>

        <div class="row">
          <div class="col-md-7">Longitude Max:</div>
          <div class="col-md-auto" id="${this.id}-maxLong"></div> 
        </div>

        <div class="row">
          <div class="col-md-7">Latitude Min:</div>
          <div class="col-md-auto" id="${this.id}-minLat"></div> 
        </div>

        <div class="row">
          <div class="col-md-7">Latitude Max:</div>
          <div class="col-md-auto" id="${this.id}-maxLat"></div> 
        </div>
      </div>
        `;
  }

  // This is called by the node.onInit() above once the HTML element for the node has been
  // created. If present, the data object may contain a number as returned by
  // Int::getData() which - if present - should be preselected.
  init(nodeDiv, data) {

    // Get our input element.
    const el = nodeDiv;

  }

  updateValues(minLong, maxLong, minLat, maxLat) {
    document.getElementById(this.id + "-minLong").innerHTML = minLong;
    document.getElementById(this.id + "-maxLong").innerHTML = maxLong;
    document.getElementById(this.id + "-minLat").innerHTML = minLat;
    document.getElementById(this.id + "-maxLat").innerHTML = maxLat;
  }
}