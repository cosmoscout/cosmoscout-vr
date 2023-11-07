////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

class RandomDataSourceComponent extends Rete.Component {

  constructor() {
    super("RandomDataSource");
    this.category = "Sources";
  }

  builder(node) {
    let output = new Rete.Output('Image2D', 'Image 2D', CosmoScout.socketTypes['Image2D']);
    node.addOutput(output)

    const boundsControl = new BoundsControl('Bounds');
    node.addControl(boundsControl);

    node.onInit = (nodeDiv) => boundsControl.init(nodeDiv, node.data);

    console.log(node.data)

    return node;
  }
}


class BoundsControl extends Rete.Control {
  constructor(key) {
    super(key);

    this.data = {
      minLat: -90,
      minLon: -180,
      maxLat: 90,
      maxLon: 180,
    };

    this.id = crypto.randomUUID();

    // This HTML code will be used whenever a node is created with this widget.
    this.template = `
      <div class="container-fluid" style="width: 200px">
        <div class="row">
          <div class="offset-2 col-3" style="text-align: center">Latitude</div>
          <div class="offset-1 col-3" style="text-align: center">Longitude</div>
        </div>
        <div class="row">
          <div class="col-2">Min:</div>
          <input id="min-lat-${this.id}" class="offset-1 col-3" type="text" value="0" style="text-align: end" />
          <input id="min-lon-${this.id}" class="offset-1 col-3" type="text" value="0" style="text-align: end" />
        </div>
        <div class="row">
          <div class="col-2">Max:</div>
          <input id="max-lat-${this.id}" class="offset-1 col-3" type="text" value="0" style="text-align: end" />
          <input id="max-lon-${this.id}" class="offset-1 col-3" type="text" value="0" style="text-align: end" />
        </div>
      </div>
    `;
  }

  init(nodeDiv, data) {
    const minLatEl = nodeDiv.querySelector(`#min-lat-${this.id}`);
    const minLonEl = nodeDiv.querySelector(`#min-lon-${this.id}`);
    const maxLatEl = nodeDiv.querySelector(`#max-lat-${this.id}`);
    const maxLonEl = nodeDiv.querySelector(`#max-lon-${this.id}`);

    if (data.minLat && data.minLon && data.maxLat && data.maxLon) {
      this.data = data;
    }

    minLatEl.value = this.data.minLat;
    minLonEl.value = this.data.minLon;
    maxLatEl.value = this.data.maxLat;
    maxLonEl.value = this.data.maxLon;

    minLatEl.addEventListener('input', e => {
      this.data.minLat = parseFloat(e.target.value);
      CosmoScout.sendMessageToCPP(this.data, this.parent.id);
    });

    minLatEl.addEventListener('pointermove', e => e.stopPropagation());

    minLonEl.addEventListener('input', e => {
      this.data.minLon = parseFloat(e.target.value);
      CosmoScout.sendMessageToCPP(this.data, this.parent.id);
    });

    minLonEl.addEventListener('pointermove', e => e.stopPropagation());

    maxLatEl.addEventListener('input', e => {
      this.data.maxLat = parseFloat(e.target.value);
      CosmoScout.sendMessageToCPP(this.data, this.parent.id);
    });

    maxLatEl.addEventListener('pointermove', e => e.stopPropagation());

    maxLonEl.addEventListener('input', e => {
      this.data.maxLon = parseFloat(e.target.value);
      CosmoScout.sendMessageToCPP(this.data, this.parent.id);
    });

    maxLonEl.addEventListener('pointermove', e => e.stopPropagation());
  }
}
