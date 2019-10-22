let animationTime = 5;
let withoutAnimationTime = 0;

//Flys the observer to a given location
function flyToLocation(planet, location, time) {
    window.call_native("fly_to", planet, location.longitude, location.latitude, location.height, time);
    window.call_native("print_notification", "Travelling", "to " + location.name, "send");
}

function parseHeight(heightStr, unit) {
    var height = parseFloat(heightStr);
    if (unit == 'mm') return height / 1000;
    else if (unit == 'cm') return height / 100;
    else if (unit == 'm') return height;
    else if (unit == 'km') return height * 1e3;
    else if (unit == 'Tsd') return height * 1e6;
    else if (unit == 'AU') return height * 1.496e11;
    else if (unit == 'ly') return height * 9.461e15;
    else if (unit == 'pc') return height * 3.086e16;

    return height * 3.086e19;
}

function parseLatitude(lat, half) {
    lat = lat.substr(0, lat.length - 1);
    if (half == 'S')
        return parseFloat(-lat);
    else
        return parseFloat(lat);
}

function parseLongitude(long, half) {
    long = long.substr(0, long.length - 1);
    if (half == 'W')
        return parseFloat(-long);
    else
        return parseFloat(long);
}

// Extracts the needed information out of the human readable place string
// and calls flyToLocation for the given location.
function geoCode(direct, planet, place, name) {
    var placeArr = place.split(" ");
    var location = {
        "longitude": parseLongitude(placeArr[0], placeArr[1]),
        "latitude": parseLatitude(placeArr[2], placeArr[3]),
        "height": parseHeight(placeArr[4], placeArr[5]),
        "name": name
    };

    if (direct) {
        flyToLocation(planet, location, withoutAnimationTime);
    } else {
        flyToLocation(planet, location, animationTime);
    }
}
