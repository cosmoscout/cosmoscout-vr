////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace operation (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// The TimeNode is pretty simple as it only has a single output socket. The component serves as
// a kind of factory. Whenever a new node is created, the builder() method is called.
class ColorMapNDComponent extends Rete.Component {

  constructor() {
    // This name must match the WCSSourceNode::sName defined in WCSSource.cpp.
    super("ColorMapND");

    // This specifies the submenu from which this node can be created in the node editor.
    this.category = "Data Extraction";
  }

  // Called whenever a new node of this type needs to be constructed.
  builder(node) {

    // This node has a single output. The first parameter is the name of this output and must be
    // unique amongst all sockets. It is also used in the WCSImageLoader::process() to write the
    // output of this node. The second parameter is shown as name on the node. The last
    // parameter references a socket type which has been registered with the node factory
    // before. It is required that the class is called <NAME>Component.

    let coverageInput =
        new Rete.Input('coverageIn', "Coverage", CosmoScout.socketTypes['Coverage']);
    node.addInput(coverageInput);

    let boundsIn = new Rete.Input('boundsIn', "Long/Lat Bounds", CosmoScout.socketTypes['RVec4']);
    node.addInput(boundsIn);

    let timeInput = new Rete.Input('wcsTimeIn', "Time", CosmoScout.socketTypes['WCSTime']);
    node.addInput(timeInput);

    let resolutionInput =
        new Rete.Input('resolutionIn', "Maximum Resolution", CosmoScout.socketTypes['Int']);
    node.addInput(resolutionInput);

    let control = new ColorWheelControl("ColorWheel");
    node.addControl(control);

    let imageOutput = new Rete.Output('imageOut', 'Image 2D', CosmoScout.socketTypes['Image2D']);
    node.addOutput(imageOutput);

    // Once the HTML element for this node has been created, the node.onInit() method will be
    // called. This is used here to initialize the widget.
    node.onInit = (nodeDiv) => {
      control.init(nodeDiv, node.data);
      node.onMessageFromCPP = (message) => { control.setData(message.data); };
    };

    return node;
  }
}

class ColorWheelControl extends Rete.Control {
  constructor(key) {
    super(key);

    this.id = crypto.randomUUID();

    // This HTML code will be used whenever a node is created with this widget.
    this.template =
        `<canvas id="${this.id}" width="400" height="400" style="margin: 0 10px"></canvas>`;
  }

  // This is called by the node.onInit() above once the HTML element for the node has been
  // created.
  init(nodeDiv, data) {

    // Get our display elements
    this.canvas = nodeDiv.querySelector('[id="' + this.id + '"]');
  }

  /**

   */
  setData(data) {
    this.drawColorWheel();
    this.drawPoints(data);
  }

  drawColorWheel() {
    const ctx     = this.canvas.getContext("2d");
    const width   = this.canvas.width;
    const height  = this.canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius  = width / 2;

    // Constant luminance
    const L         = 0.7; // between 0 (black) and 1 (white)
    const maxChroma = 0.3; // Adjust for safe displayable colors

    const imageData = ctx.createImageData(width, height);
    const pixels    = imageData.data;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const dx   = x - centerX;
        const dy   = y - centerY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > radius)
          continue;

        const angle = Math.atan2(dy, dx);
        const h     = ((angle * 180) / Math.PI + 360) % 360;
        const c     = (dist / radius) * maxChroma;

        // Construct oklch color and convert to RGB
        let color = chroma.oklch(L, c, h).rgb();

        const index       = (y * width + x) * 4;
        pixels[index]     = color[0];
        pixels[index + 1] = color[1];
        pixels[index + 2] = color[2];
        pixels[index + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }

  drawPoints(data) {
    const ctx     = this.canvas.getContext("2d");
    const width   = this.canvas.width;
    const height  = this.canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius  = width / 2;

    ctx.fillStyle   = "black";
    ctx.strokeStyle = "white";
    ctx.lineWidth   = 1;

    data.forEach((point) => {
      const x = point[0] * radius + centerX;
      const y = point[1] * radius + centerY;

      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  }
}