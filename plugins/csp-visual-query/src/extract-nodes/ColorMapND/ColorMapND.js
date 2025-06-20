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
    this.template = `<div id="${
        this.id}" class="colorMapND" style="position: relative; width: 400px; height: 400px">
          <canvas class="color-layer" style="position: absolute; top: 50px; left: 50px;" width=300 height=300></canvas>
          <canvas class="point-layer" style="position: absolute; top: 50px; left: 50px;" width=300 height=300></canvas>
          <div    class="label-layer" style="position: absolute; width: 400px; height: 400px"></div>
        </div>`;
  }

  // This is called by the node.onInit() above once the HTML element for the node has been
  // created.
  init(nodeDiv, data) {

    // Get our display elements
    this.container   = nodeDiv.querySelector('[id="' + this.id + '"]');
    this.labels      = this.container.querySelector(".label-layer");
    this.pointCanvas = this.container.querySelector(".point-layer");
    this.colorCanvas = this.container.querySelector(".color-layer");

    this.drawColorWheel();
  }

  /**

   */
  setData(data) {
    this.drawDimensions(data.dimensions);
    this.drawPoints(data.points);
  }

  drawColorWheel() {
    const ctx     = this.colorCanvas.getContext("2d");
    const width   = this.colorCanvas.width;
    const height  = this.colorCanvas.height;
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

  drawPoints(points) {

    this.points = points || this.points;

    const directions =
        this.dimensions.map((angle) => { return [Math.cos(angle), Math.sin(angle)]; });

    const ctx     = this.pointCanvas.getContext("2d");
    const width   = this.pointCanvas.width;
    const height  = this.pointCanvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius  = width / 2;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle   = "black";
    ctx.strokeStyle = "white";
    ctx.lineWidth   = 1;

    this.points.forEach((point) => {
      const position  = [0, 0];
      let   weightSum = 0;

      // Compute the position of the point in the color wheel.
      for (let i = 0; i < this.dimensions.length; i++) {
        const weight = point[i] ** 2;
        weightSum += weight;
        position[0] += weight * directions[i][0];
        position[1] += weight * directions[i][1];
      }

      // If the point has no weight, skip it.
      if (weightSum === 0) {
        return;
      }

      const x = position[0] * radius / weightSum + centerX;
      const y = position[1] * radius / weightSum + centerY;

      ctx.beginPath();
      ctx.arc(x, y, 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
  }

  drawDimensions(dimensions) {

    this.dimensions = dimensions || this.dimensions;

    // Remove all children from the canvas
    this.labels.innerHTML = "";

    // Add a div for each dimension.
    this.dimensions.forEach((angle, i) => {
      const div       = document.createElement("div");
      div.className   = "dimension-label";
      div.textContent = i;
      div.style.transform =
          `translate(-50%, -50%) rotate(${angle}rad) translate(170px, 0) rotate(90deg)`;

      // Make the label draggable around the center.
      div.addEventListener("pointerdown", (event) => {
        // Compute the center of the container.
        const rect    = this.container.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;

        event.preventDefault();
        event.stopPropagation();

        const onPointerMove = (moveEvent) => {
          const dx    = moveEvent.clientX - centerX;
          const dy    = moveEvent.clientY - centerY;
          const angle = Math.atan2(dy, dx);
          div.style.transform =
              `translate(-50%, -50%) rotate(${angle}rad) translate(170px, 0) rotate(90deg)`;

          this.dimensions[i] = angle;

          this.drawPoints();

          event.stopPropagation();
          event.preventDefault();
        };

        const onPointerUp = () => {
          CosmoScout.sendMessageToCPP(
              {operation: "setDimensions", dimensions: this.dimensions}, this.parent.id);
          document.removeEventListener("pointermove", onPointerMove);
          document.removeEventListener("pointerup", onPointerUp);
        };

        document.addEventListener("pointermove", onPointerMove);
        document.addEventListener("pointerup", onPointerUp);
      });

      this.labels.appendChild(div);
    });
  }
}