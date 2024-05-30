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
    this.category = "Input";
  }

  // Called whenever a new node of this type needs to be constructed.
  builder(node) {

    // This node has a single output. The first parameter is the name of this output and must be
    // unique amongst all sockets. It is also used in the WCSImageLoader::process() to write the
    // output of this node. The second parameter is shown as name on the node. The last
    // parameter references a socket type which has been registered with the node factory
    // before. It is required that the class is called <NAME>Component.

    let imageOutput =
        new Rete.Output('coverageOut', 'Coverage', CosmoScout.socketTypes['Coverage']);
    node.addOutput(imageOutput);

    let timeIntervals = new Rete.Output(
        'timeIntervalsOut', 'Time Intervals', CosmoScout.socketTypes['WCSTimeIntervals']);
    node.addOutput(timeIntervals);

    let boundsOut =
        new Rete.Output('boundsOut', "Long/Lat Bounds", CosmoScout.socketTypes['RVec4']);
    node.addOutput(boundsOut);

    let serverDropDown = new DropDownControl('selectServer',
        (newServer) => { CosmoScout.sendMessageToCPP({server: newServer.text}, node.id); },
        "Server", [{value: 0, text: "None"}]);
    node.addControl(serverDropDown);

    let coverageDropDown = new DropDownControl('selectCoverage', (newCoverage) => {
      CosmoScout.sendMessageToCPP({imageChannel: newCoverage.text}, node.id);
    }, "Coverage", [{value: 0, text: "None"}]);
    node.addControl(coverageDropDown);

    node.onMessageFromCPP = (message) => {
      if (message["servers"]) {
        const servers = message["servers"].map((server, index) => ({value: index, text: server}));
        node.data.servers = servers;
        serverDropDown.setOptions(servers);
      }

      // new image channels received
      else if (message["imageChannels"]) {

        // rest image channels selection
        if (message["imageChannels"] === "reset") {
          node.data.imageChannels = [];
          coverageDropDown.setOptions([{value: 0, text: "None"}]);
        } else {
          const coverages =
              message["imageChannels"].map((channel, index) => ({value: index, text: channel}));
          node.data.coverages = coverages;
          coverageDropDown.setOptions(coverages);
        }
      } else {
        console.log("Unknown cpp message:");
        console.log(message);
      }
    };

    node.onInit = (nodeDiv) => {
      serverDropDown.init(nodeDiv, {
        options: node.data.servers?.map((server, index) => ({value: index, text: server})),
        selectedValue: node.data.selectedServer
      });

      coverageDropDown.init(nodeDiv, {
        options: node.data.coverages?.map((channel, index) => ({value: index, text: channel})),
        selectedValue: node.data.selectedCoverage
      });
    };

    return node;
  }
}