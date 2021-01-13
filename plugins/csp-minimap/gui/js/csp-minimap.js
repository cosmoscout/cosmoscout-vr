/* global IApi, CosmoScout */

(() => {
  /**
   * Minimap Api
   */
  class MinimapApi extends IApi {
    /**
     * @inheritDoc
     * @type {string}
     */
    name = 'minimap';

    _bookmarks = {};
    _baselayer = null;

    _lastLng = 0;
    _lastLat = 0;
    _locked  = true;

    _customControls = L.Control.extend({
      options: {position: 'topleft'},
      onAdd:
          () => {
            let createButton = (icon, tooltip, callback) => {
              var button       = L.DomUtil.create('a');
              button.innerHTML = `<i class="material-icons">${icon}</i>`;
              button.setAttribute('title', tooltip);
              button.dataset.toggle    = 'tooltip';
              button.dataset.placement = 'left';
              L.DomEvent.on(button, 'click', callback);
              L.DomEvent.on(button, 'dblclick', L.DomEvent.stop);
              L.DomEvent.on(button, 'mousedown', L.DomEvent.stop);
              L.DomEvent.on(button, 'mouseup', L.DomEvent.stop);

              return button;
            };

            let zoomInButton = createButton('add', 'Zoom in', (e) => {
              this._map.zoomIn();
              L.DomEvent.stop(e);
            });

            let zoomOutButton = createButton('remove', 'Zoom out', (e) => {
              this._map.zoomOut();
              L.DomEvent.stop(e);
            });

            let zoomResetButton = createButton('zoom_out_map', 'Fit map', (e) => {
              this._map.setView([0, 0], 0);
              L.DomEvent.stop(e);
            });

            let lockButton = createButton('lock', 'Lock Minimap to Observer', (e) => {
              this._locked = !this._locked;
              if (this._locked) {
                e.target.querySelector('i').textContent = 'lock';
              } else {
                e.target.querySelector('i').textContent = 'lock_open';
              }

              L.DomEvent.stop(e);
            });

            let centerButton = createButton('my_location', 'Center Minimap on Observer', (e) => {
              this._centerMapOnObserver(true);
              L.DomEvent.stop(e);
            });

            let bookmarkButton = createButton('bookmark', 'Add a Bookmark', (e) => {
              CosmoScout.bookmarkEditor.addNewBookmark();
              L.DomEvent.stop(e);
            });

            let container = L.DomUtil.create('div');

            let groupA = L.DomUtil.create('div');
            groupA.classList.add('leaflet-bar')
            groupA.appendChild(zoomInButton);
            groupA.appendChild(zoomOutButton);
            groupA.appendChild(zoomResetButton);
            container.appendChild(groupA);

            let groupB = L.DomUtil.create('div');
            groupB.classList.add('leaflet-bar')
            groupB.style.marginTop = "10px";
            groupB.appendChild(centerButton);
            groupB.appendChild(lockButton);
            container.appendChild(groupB);

            let groupC = L.DomUtil.create('div');
            groupC.classList.add('leaflet-bar')
            groupC.style.marginTop = "10px";
            groupC.appendChild(bookmarkButton);
            container.appendChild(groupC);

            return container;
          }
    });

    init() {
      // Add the minimap window.
      this._mapDiv = CosmoScout.gui.loadTemplateContent('minimap');
      document.getElementById('cosmoscout').appendChild(this._mapDiv);

      // Create the Leaflet map.
      this._map = L.map(document.querySelector('#minimap .window-content'), {
        attributionControl: false,
        zoomControl: false,
        center: [0, 0],
        zoom: 0,
        worldCopyJump: true,
        maxBounds: [[-90, -180], [90, 180]],
        crs: L.CRS.EPSG4326
      });

      // Bookmarks will be shown in this cluster layer.
      this._bookmarkLayer = L.markerClusterGroup({
        spiderfyOnMaxZoom: true,
        showCoverageOnHover: false,
        zoomToBoundsOnClick: true,
        animateAddingMarkers: true,
        maxClusterRadius: 40,
        iconCreateFunction: (cluster) => {
          return L.divIcon({
            className: 'minimap-bookmark-cluster',
            html: '<div>' + cluster.getChildCount() + '</div>',
            iconSize: [22, 22],
            iconAnchor: [11, 25]
          });
        }
      });

      // Add our custom buttons.
      this._map.addControl(new this._customControls());

      // Add the attribution control to the bottom left.
      L.control.attribution({position: 'bottomleft', prefix: false}).addTo(this._map);

      // Create a marker for the user's position.
      let crossHair = L.icon({
        iconUrl: 'img/observer.png',
        shadowUrl: 'img/observer_shadow.png',
        iconSize: [36, 36],
        shadowSize: [36, 36],
        iconAnchor: [18, 18],
        shadowAnchor: [18, 18]
      });

      this._observerMarker =
          L.marker(
               [0, 0], {icon: crossHair, interactive: false, keyboard: false, zIndexOffset: -100})
              .addTo(this._map);

      // Move the observer when clicked on the minimap.
      this._map.on('click', (e) => {
        let lng    = parseFloat(e.latlng.lng);
        let lat    = parseFloat(e.latlng.lat);
        let center = CosmoScout.state.activePlanetCenter;
        let height = this._zoomToHeight(this._map.getZoom());

        if (center !== "") {
          CosmoScout.callbacks.navigation.setBodyLongLatHeightDuration(center, lng, lat, height, 2);
        }
      });

      // Resize the leaflet map on minimap window resize events.
      this._resizeObserver = new ResizeObserver((entries) => {
        // We wrap it in requestAnimationFrame to avoid "ResizeObserver loop limit exceeded".
        // See https://stackoverflow.com/questions/49384120/resizeobserver-loop-limit-exceeded
        window.requestAnimationFrame(() => {
          if (!Array.isArray(entries) || !entries.length) {
            return;
          }
          this._map.invalidateSize();
        });
      });

      this._resizeObserver.observe(this._mapDiv);
    }

    // Update minimap based on observer state.
    update() {
      if (this._mapDiv.classList.contains("visible")) {
        this._centerMapOnObserver(false);
      }
    }

    configure(settingsJSON) {

      // First remove existing layer.
      if (this._baselayer) {
        this._map.removeLayer(this._baselayer);
      }

      if (settingsJSON == "") {

        // Add empty layer if none is configured.
        this._baselayer = L.tileLayer('');

      } else {

        let settings = JSON.parse(settingsJSON);

        // Update map projection.
        if (settings.projection == "mercator") {
          this._map.options.crs = L.CRS.EPSG3857;
        } else {
          this._map.options.crs = L.CRS.EPSG4326;
        }

        // Create new layer.
        if (settings.type == "wms") {
          this._baselayer = L.tileLayer.wms(settings.url, settings.config);
        } else {
          this._baselayer = L.tileLayer(settings.url, settings.config);
        }
      }

      // Now add the new layer.
      this._map.addLayer(this._baselayer);
      this._map.setZoom(2);
    }

    addBookmark(bookmarkID, bookmarkColor, lng, lat) {
      this._bookmarks[bookmarkID] =
          L.marker([lat, lng], {
             icon: L.divIcon({
               className: 'minimap-bookmark-icon',
               html: `<div style="background-color: ${bookmarkColor}"></div>`,
               iconSize: [22, 22],
               iconAnchor: [11, 26],
             })
           }).addTo(this._bookmarkLayer);

      L.DomEvent.on(this._bookmarks[bookmarkID], 'mouseover', () => {
        let pos = this._map.latLngToContainerPoint(this._bookmarks[bookmarkID].getLatLng());
        let box = this._mapDiv.getBoundingClientRect();
        CosmoScout.callbacks.bookmark.showTooltip(
            bookmarkID, box.x + pos.x + 2, box.y + pos.y - 12);
      });

      L.DomEvent.on(this._bookmarks[bookmarkID], 'mouseout', () => {
        CosmoScout.callbacks.bookmark.hideTooltip();
      });

      L.DomEvent.on(this._bookmarks[bookmarkID], 'click', (e) => {
        CosmoScout.callbacks.bookmark.gotoLocation(bookmarkID);
        L.DomEvent.stop(e);
      });

      // The markerClusterGroup wants to be added to the map AFTER the first marker has been added.
      // So we have to do this here...
      if (!this._bookmarkLayerAdded) {
        this._map.addLayer(this._bookmarkLayer);
        this._bookmarkLayerAdded = true;
      }
    }

    removeBookmark(bookmarkID) {
      if (bookmarkID in this._bookmarks) {
        this._bookmarks[bookmarkID].removeFrom(this._bookmarkLayer);
        delete this._bookmarks[bookmarkID];
      }
    }

    removeBookmarks() {
      for (let i in this._bookmarks) {
        this._bookmarks[i].removeFrom(this._bookmarkLayer);
      }

      this._bookmarks = {};
    }

    // These are quite a crude conversions from the minimap zoom level to observer height. It
    // works quite well for middle latitudes and the standard field of view.
    _zoomToHeight(zoom) {
      return 0.01 * this._mapDiv.clientWidth * CosmoScout.state.activePlanetRadius[0] /
             Math.pow(2, zoom);
    }

    _heightToZoom(height) {
      return Math.log2(0.01 * this._mapDiv.clientWidth * CosmoScout.state.activePlanetRadius[0] /
                       Math.max(100, height));
    }

    // Update minimap based on observer state.
    _centerMapOnObserver(force) {
      if (CosmoScout.state.observerLngLatHeight) {

        // Update position of observer marker.
        let [lng, lat] = CosmoScout.state.observerLngLatHeight;

        // Center the minimap around the observer.
        if (force || (this._lastLng !== lng || this._lastLat !== lat)) {
          this._observerMarker.setLatLng([lat, lng]);

          if (force || this._locked) {
            this._map.setView([lat, lng], [this._map.getZoom()]);
          }

          this._lastLng = lng;
          this._lastLat = lat;
        }
      }
    }
  }

  CosmoScout.init(MinimapApi);
})();
