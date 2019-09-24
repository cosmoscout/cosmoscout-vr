
//Flys the observer to a given location
function flyToLocation(planet, location) {
        window.call_native("fly_to", planet, location.longitude, location.latitude, location.height);
        window.call_native("print_notification", "Travelling", "to " + location.name, "send");
}

// Gets the geo code for a location and then calls flyToLocation
function geo_code(planet, place) {

    if (planet === "Earth")
    {
        $.ajax({
            url: "https://nominatim.openstreetmap.org/search?q="+ encodeURIComponent(place) +"&format=json&limit=1",
            type: 'GET',
            dataType: 'json',
            success: function(data) {
                if (data.length === 0)
                {
                    window.call_native("print_notification", "Error", "Location not found!", "error");
                    return;
                }

                var bounds = data[0].boundingbox;
                var lat = bounds[1] - bounds[0];
                var lon = bounds[3] - bounds[2];

                var location = {
                    "latitude":  parseFloat(data[0].lat),
                    "longitude": parseFloat(data[0].lon),
                    "height":    Math.max(lat, lon) * 111 * 1000,
                    "name":      data[0].display_name
                };

                flyToLocation(planet, location);
            },
            error: function() {
                console.log("Error requesting Data from openstreetmap");
            }
        });
    } else {
        planetLowerCase = planet.toLowerCase();
        if (!locations[planetLowerCase]) {
            window.call_native("print_notification", "Error", "No location for " +planet + "!", "error");
            return;
        }

        var fuzzyset = FuzzySet(Object.keys(locations[planetLowerCase]));
        var name = fuzzyset.get(place)[0][1];
        var location = locations[planetLowerCase][name];
        var height = location[0] == 0 ? 10000 : location[0]*2000;

        var location = {
            "latitude":  location[1],
            "longitude": location[2],
            "height":    height,
            "name":      name
        };

        flyToLocation(planet, location);
    }
}
