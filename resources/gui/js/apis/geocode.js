/* global IApi, CosmoScout */

/**
 * The geo-coding API. This does not require any other CosmoScout APIs, so you can use it also in
 * other UI elements, not only in the main CosmoScout UI.
 */
class GeoCodeApi extends IApi {
  /**
   * @inheritDoc
   */
  name = 'geocode';

  _locations = {};

  /**
   * For Earth, this uses OpenStreetMap, for other planets a CSV file in ../share/locations is used.
   * This requires third-party/js/fuzzyset.js to be included.
   *
   * @param {string}   planet   The planet to perform geocoding for.
   * @param {string}   query    A name of a location to get coordinates for.
   * @param {function} callback The callback will receive one parameter, an object containing a
   *                            latitude, a longitude, a diameter and a name. If no location was
   *                            found, the parameter will be undefined.
   */
  forward(planet, query, callback) {
    if (planet.toLowerCase() === "earth") {
      $.ajax({
        url: "https://nominatim.openstreetmap.org/search?q=" + encodeURIComponent(query) +
                 "&format=json&limit=1",
        type: 'GET',
        dataType: 'json',
        success: (data) => {
          if (data.length === 0) {
            callback();
            return;
          }

          let bounds   = data[0].boundingbox;
          let posA     = this._toCartesian(bounds[2], bounds[0]);
          let posB     = this._toCartesian(bounds[3], bounds[1]);
          let diameter = this._dist(posA, posB) * 6371;

          callback({
            "name": data[0].display_name,
            "longitude": parseFloat(data[0].lon),
            "latitude": parseFloat(data[0].lat),
            "diameter": diameter
          });
        },
        error: function() {
          console.log("Error requesting Data from OpenStreetMap!");
          callback();
        },
        beforeSend: (xhr) => xhr.setRequestHeader('Access-Control-Allow-Origin', '*')
      });
    } else {

      this._getLocationsCSV(planet, (locations) => {
        if (!locations) {
          callback();
          return;
        }

        let fuzzyset = FuzzySet(Object.keys(locations));
        let name     = fuzzyset.get(query)[0][1];

        let location = locations[name];
        let diameter = location[0] == 0 ? 1 : location[0];

        callback({
          "name": name,
          "longitude": location[2],
          "latitude": location[1],
          "diameter": diameter
        });
      });
    }
  }

  /**
   * For Earth, this uses OpenStreetMap, for other planets a CSV file in ../share/locations is used.
   *
   * @param {string}   planet    The planet to perform geocoding for.
   * @param {number}   longitude The geographic longitude.
   * @param {number}   latitude  The geographic latitude.
   * @param {function} callback  The callback will receive one parameter, a line of text describing
   *                             the location. If no location was found, the parameter will be
   *                             undefined.
   */
  reverse(planet, longitude, latitude, callback) {
    if (planet.toLowerCase() === "earth") {
      $.ajax({
        url: "https://nominatim.openstreetmap.org/reverse?format=json&lat=" + latitude +
                 "&lon=" + longitude,
        type: 'GET',
        dataType: 'json',
        success: function(data) {
          let result = planet;
          let a      = data.address;

          if (a) {
            if (a.country) {
              result = a.country;
            }

            if (a.city) {
              result = a.city + ", " + result;
            } else if (a.town) {
              result = a.town + ", " + result;
            }

            let street = a.road;

            if (!street) {
              street = a.pedestrian;
            }

            if (street) {
              if (a.house_number) {
                result = street + " " + a.house_number + ", " + result;
              } else {
                result = street + ", " + result;
              }
            }
          }

          callback(result);
        },
        error: function() {
          console.log("Error requesting Data from OpenStreetMap!");
          callback();
        },
        beforeSend: (xhr) => xhr.setRequestHeader('Access-Control-Allow-Origin', '*')
      });
    } else {

      this._getLocationsCSV(planet, (locations) => {
        if (!locations) {
          callback();
          return;
        }

        let bestScore = 0;
        let result    = planet;
        let queryPos  = this._toCartesian(longitude, latitude);

        Object.keys(locations).forEach((name) => {
          let loc      = locations[name];
          let pos      = this._toCartesian(loc[2], loc[1]);
          let dist     = this._dist(pos, queryPos);
          let diameter = loc[0] > 0 ? loc[0] : 1.0;
          let score    = Math.max(0.0, 1.0 - dist / diameter);

          if (score > bestScore) {
            bestScore = score;
            result    = name;
          }
        });

        callback(result);
      });
    }
  }

  /**
   * Loads location data from a file in ../share/locations/.
   *
   * @param {string} planet     SPICE center name of the planet to search for a locations file.
   * @param {function} callback An object containing locations will be passed as parameter. The
   *                            parameter will be undefined on error.
   * @private
   */
  _getLocationsCSV(planet, callback) {
    planet = planet.toLowerCase();
    if (this._locations[planet]) {
      callback(this._locations[planet]);
    } else {
      $.ajax({
        type: "GET",
        url: `../locations/${planet}.csv`,
        dataType: "text",
        error: ()       => callback(),
        success: (data) => {
          this._locations[planet] = {};
          let lines               = data.split('\n');
          for (let i = lines.length - 1; i > 0; --i) {
            let elems = lines[i].split(',');
            let name  = elems[1].replace(/^"(.*)"$/, '$1'); // Remove "" from names
            this._locations[planet][name] = [
              parseFloat(elems[2]), // The diameter
              parseFloat(elems[3]), // The latitude
              parseFloat(elems[4])  // The longitude
            ];
          }
          callback(this._locations[planet]);
        }
      });
    }
  }

  /**
   * Converts the given location to cartesian coordinates on the unit sphere.
   *
   * @param {number} longitude In degrees.
   * @param {number} latitude  In degrees.
   * @private
   */
  _toCartesian(longitude, latitude) {
    const lon = longitude * Math.PI / 180.0;
    const lat = latitude * Math.PI / 180.0;

    const x = Math.cos(lat) * Math.sin(lon);
    const y = Math.sin(lat);
    const z = Math.cos(lat) * Math.cos(lon);

    return [x, y, z];
  }

  /**
   * Computes the cartesian distance between two 3D-points.
   *
   * @param {number} a
   * @param {number} b
   * @private
   */
  _dist(a, b) {
    return Math.sqrt(
        Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2) + Math.pow(a[2] - b[2], 2));
  }
}
