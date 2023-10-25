////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// The TimeNode is pretty simple as it only has a single output socket. The component serves as
// a kind of factory. Whenever a new node is created, the builder() method is called.
class WCSCoverageComponent extends Rete.Component {

  constructor() {
    // This name must match the WCSSourceNode::sName defined in WCSSource.cpp.
    super("WCSCoverage");

    // This specifies the submenu from which this node can be created in the node editor.
    this.category = "Sources";
  }

  // Called whenever a new node of this type needs to be constructed.
  builder(node) {

    // This node has a single output. The first parameter is the name of this output and must be
    // unique amongst all sockets. It is also used in the WCSImageLoader::process() to write the
    // output of this node. The second parameter is shown as name on the node. The last
    // parameter references a socket type which has been registered with the node factory
    // before. It is required that the class is called <NAME>Component.

    let imageOutput = new Rete.Output('coverageOut', 'Coverage', CosmoScout.socketTypes['Coverage']);
    node.addOutput(imageOutput);
    
    let minTimeOutput = new Rete.Output('minTimeValueOut', 'Min Time', CosmoScout.socketTypes['number Value']);
    node.addOutput(minTimeOutput);

    let maxTimeOutput = new Rete.Output('maxTimeValueOut', 'Max Time', CosmoScout.socketTypes['number Value']);
    node.addOutput(maxTimeOutput);

    let lngBoundMinOutput = new Rete.Output('lngBoundMinOut', "Longitude Min", CosmoScout.socketTypes['number Value']);
    node.addOutput(lngBoundMinOutput);

    let lngBoundMaxOutput = new Rete.Output('lngBoundMaxOut', "Longitude Max", CosmoScout.socketTypes['number Value']);
    node.addOutput(lngBoundMaxOutput);

    let latBoundMinOutput = new Rete.Output('latBoundMinOut', "Latitude Min", CosmoScout.socketTypes['number Value']);
    node.addOutput(latBoundMinOutput);

    let latBoundMaxOutput = new Rete.Output('latBoundMaxOut', "Latitude Max", CosmoScout.socketTypes['number Value']);
    node.addOutput(latBoundMaxOutput);

    let serverControl = new ServerControl('selectServer');
    node.addControl(serverControl);

    let imageChannelControl = new ImageChannelControl('selectImageChannel');
    node.addControl(imageChannelControl);

    node.onInit = (nodeDiv) => { 

      serverControl.init(nodeDiv, node.data); 
      imageChannelControl.init(nodeDiv, node.data);

      node.onMessageFromCPP = (message) => {

        // display servers in dropdown
        if (message["server"]) {
          node.data.url = message["server"];
          serverControl.createServerSelection(message["server"]);  
        }

        // new image channels received
        else if (message["imageChannel"]) {

          // rest image channels selection
          if (message["imageChannel"] === "reset") {
            node.data.imageChannel = [];
            imageChannelControl.resetImageChannelSelection();

          // add new image channels to dropdown
          } else {
            node.data.imageChannel = message["imageChannel"];
            imageChannelControl.createImageChannelSelection(message["imageChannel"]);
          }
        
        } else {
          console.log("Unknown cpp message:");
          console.log(message);
        }
      };
    }

    let button1 = document.createElement("button");
    button1.innerHTML = "request server";
    button1.addEventListener("click", (e) => {
      serverControl.sendServerRequest();
    });
    document.body.insertBefore(button1, document.body.firstChild);

    return node;
  }
}

// This is the widget which is used for selecting the server.
class ServerControl extends Rete.Control {
  constructor(key) {
    super(key);
    this.selectElement;

    // This HTML code will be used whenever a node is created with this widget.
    this.template = `
          <select>
            <option value="none">None</option>
          </select>

          <style>
            .dropdown {
              margin: 10px 15px !important;
              width: 150px !important;
            }
          </style>
        `;
  }

  // This is called by the node.onInit() above once the HTML element for the node has been
  // created. If present, the data object may contain a math operation as returned by
  // MathNode::getData() which - if present - should be preselected.
  init(nodeDiv, data) {

    // Initialize the bootstrap select.
    this.selectElement = nodeDiv.querySelector("select");
    $(this.selectElement).selectpicker();

    // Preselect a server.
    if (data.url) {
      $(this.selectElement).selectpicker('val', data.url);
    }

    // Send an update to the node editor server whenever the user selects a new server.
    this.selectElement.addEventListener('change',
      (e) => {
        CosmoScout.sendMessageToCPP({server: e.target.value}, this.parent.id);
      });
  }

  // Send a request to get the available servers.
  sendServerRequest() {
    if (!this.parent.data.url) {
      CosmoScout.sendMessageToCPP("requestServers", this.parent.id);
    }
  }

  createServerSelection(values) {
    // add new elements
    for (let i = 0; i < values.length; i++) {
      let option = document.createElement("option");
      option.value = values[i];
      option.innerHTML = values[i];
      this.selectElement.appendChild(option);
    }
    // refresh bootstrap dropdown options
    $(this.selectElement).selectpicker("refresh");     
  }
}

// This is the widget which is used for selecting the image Channel.
class ImageChannelControl extends Rete.Control {
  constructor(key) {
    super(key);
    this.selectElement;

    // This HTML code will be used whenever a node is created with this widget.
    this.template = `
          <select>
            <option value="0">None</option>
          </select>

          <style>
            .dropdown {
              margin: 10px 15px !important;
              width: 150px !important;
            }
          </style>
        `;
  }

  // This is called by the node.onInit() above once the HTML element for the node has been
  // created. If present, the data object may contain a math operation as returned by
  // MathNode::getData() which - if present - should be preselected.
  init(nodeDiv, data) {

    // Initialize the bootstrap select.
    this.selectElement = nodeDiv.querySelectorAll("select")[1];

    $(this.selectElement).selectpicker();

    // Preselect a server.
    if (data.imageChannel) {
      $(this.selectElement).selectpicker('val', data.imageChannel);
    }

    // Send an update to the node editor server whenever the user selects a new channel.
    this.selectElement.addEventListener('change',
      (e) => {
        CosmoScout.sendMessageToCPP({imageChannel: e.target.value}, this.parent.id);
      });
  }

  createImageChannelSelection(values) {
    this.resetImageChannelSelection();

    // add new elements
    for (let i = 0; i < values.length; i++) {
      let option = document.createElement("option");
      option.value = values[i];
      option.innerHTML = values[i];
      this.selectElement.appendChild(option);
    }
    // refresh bootstrap dropdown options
    $(this.selectElement).selectpicker("refresh");
  }

  resetImageChannelSelection() {
    // remove all elements
    while (this.selectElement.firstChild) {
      this.selectElement.removeChild(this.selectElement.lastChild);
    }

    // add "None" element
    let none = document.createElement("option");
    none.value = 0;
    none.innerHTML = "None";
    this.selectElement.appendChild(none);

    // refresh bootstrap dropdown options
    $(this.selectElement).selectpicker("refresh");
  }
}