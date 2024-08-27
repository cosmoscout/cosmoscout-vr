////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

class RandomDataSource3DComponent extends Rete.Component {

  constructor() {
    super("RandomDataSource3D");
    this.category = "Sources";
  }

  builder(node) {
    let output = new Rete.Output('Volume3D', 'Volume 3D', CosmoScout.socketTypes['Volume3D']);
    node.addOutput(output)

    const boundsControl = new BoundsControl3D('Bounds3D');
    node.addControl(boundsControl);

    node.onInit = (nodeDiv) => boundsControl.init(nodeDiv, node.data);

    console.log(node.data)

    return node;
  }
}


class BoundsControl3D extends Rete.Control {
  constructor(key) {
    super(key);

    this.data = {
      minLat: -90,
      minLon: -180,
      maxLat: 90,
      maxLon: 180,
      minHeight: 0,
      maxHeight: 100_000,
    };

    this.id = crypto.randomUUID();

    // This HTML code will be used whenever a node is created with this widget.
    this.template = `
      <div class="container-fluid" style="width: 300px">
        <div class="row">
          <div class="offset-3 col-3" style="text-align: center">Min</div>
          <div class="offset-1 col-3" style="text-align: center">Max</div>
        </div>
        <div class="row">
          <div class="col-2">Lat:</div>
          <input id="min-lat-${this.id}" class="offset-1 col-3" type="text" value="0" style="text-align: end" />
          <input id="max-lat-${this.id}" class="offset-1 col-3" type="text" value="0" style="text-align: end" />
        </div>
        <div class="row">
          <div class="col-2">Lon:</div>
          <input id="min-lon-${this.id}" class="offset-1 col-3" type="text" value="0" style="text-align: end" />
          <input id="max-lon-${this.id}" class="offset-1 col-3" type="text" value="0" style="text-align: end" />
        </div>
        <div class="row">
          <div class="col-2">Height:</div>
          <input id="min-height-${this.id}" class="offset-1 col-3" type="text" value="0" style="text-align: end" />
          <input id="max-height-${this.id}" class="offset-1 col-3" type="text" value="0" style="text-align: end" />
        </div>
      </div>
    `;
  }

  init(nodeDiv, data) {
    const minLatEl = nodeDiv.querySelector(`#min-lat-${this.id}`);
    const maxLatEl = nodeDiv.querySelector(`#max-lat-${this.id}`);
    const minLonEl = nodeDiv.querySelector(`#min-lon-${this.id}`);
    const maxLonEl = nodeDiv.querySelector(`#max-lon-${this.id}`);
    const minHeightEl = nodeDiv.querySelector(`#min-height-${this.id}`);
    const maxHeightEl = nodeDiv.querySelector(`#max-height-${this.id}`);

    if (typeof data.minLat == 'number' && typeof data.maxLat == 'number' && typeof data.minLon == 'number' &&
      typeof data.maxLon == 'number' && typeof data.minHeight == 'number' && typeof data.maxHeight == 'number') {
      this.data = data;
    }

    minLatEl.value    = this.data.minLat;
    maxLatEl.value    = this.data.maxLat;
    minLonEl.value    = this.data.minLon;
    maxLonEl.value    = this.data.maxLon;
    minHeightEl.value = this.data.minHeight;
    maxHeightEl.value = this.data.maxHeight;

    minLatEl.addEventListener('input', e => {
      this.data.minLat = parseFloat(e.target.value);
      CosmoScout.sendMessageToCPP(this.data, this.parent.id);
    });

    minLatEl.addEventListener('pointermove', e => e.stopPropagation());

    maxLatEl.addEventListener('input', e => {
      this.data.maxLat = parseFloat(e.target.value);
      CosmoScout.sendMessageToCPP(this.data, this.parent.id);
    });

    maxLatEl.addEventListener('pointermove', e => e.stopPropagation());

    minLonEl.addEventListener('input', e => {
      this.data.minLon = parseFloat(e.target.value);
      CosmoScout.sendMessageToCPP(this.data, this.parent.id);
    });

    minLonEl.addEventListener('pointermove', e => e.stopPropagation());

    maxLonEl.addEventListener('input', e => {
      this.data.maxLon = parseFloat(e.target.value);
      CosmoScout.sendMessageToCPP(this.data, this.parent.id);
    });

    maxLonEl.addEventListener('pointermove', e => e.stopPropagation());

    minHeightEl.addEventListener('input', e => {
      this.data.minHeight = parseFloat(e.target.value);
      CosmoScout.sendMessageToCPP(this.data, this.parent.id);
    });

    minHeightEl.addEventListener('pointermove', e => e.stopPropagation());

    maxHeightEl.addEventListener('input', e => {
      this.data.maxHeight = parseFloat(e.target.value);
      CosmoScout.sendMessageToCPP(this.data, this.parent.id);
    });

    maxHeightEl.addEventListener('pointermove', e => e.stopPropagation());
  }
}
