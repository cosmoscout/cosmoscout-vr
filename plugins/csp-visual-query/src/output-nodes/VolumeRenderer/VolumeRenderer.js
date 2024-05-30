////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

class VolumeRendererComponent extends Rete.Component {
  constructor() {
    // This name must match the RenderNode::sName defined in Render.cpp.
    super("VolumeRenderer");

    // This specifies the submenu from which this node can be created in the node editor.
    this.category = "Output";
  }

  // Called whenever a new node of this type needs to be constructed.
  builder(node) {
    let input = new Rete.Input('Volume3D', "Volume 3D", CosmoScout.socketTypes['Volume3D']);
    node.addInput(input);

    let lutInput = new Rete.Input('lut', "LookUp Table", CosmoScout.socketTypes['LUT']);
    node.addInput(lutInput);

    const dropDownCallback = (selection) => CosmoScout.sendMessageToCPP(selection, node.id);

    let centerControl =
        new DropDownControl('center', dropDownCallback, "Body", [{value: 0, text: 'None'}]);
    node.addControl(centerControl);

    node.onMessageFromCPP = (message) => centerControl.setOptions(
        message.map((centerName, index) => ({value: index, text: centerName})));

    node.onInit =
        (nodeDiv) => {
          centerControl.init(nodeDiv, {
            options: node.data.options?.map((body, index) => ({value: index, text: body})),
            selectedValue: node.data.selectedBody
          });
        }

    return node;
  }
}
