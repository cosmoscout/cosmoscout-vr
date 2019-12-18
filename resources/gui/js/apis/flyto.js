class FlyToApi extends IApi {
    name = 'flyto';

    flyTo(planet, location, time) {
      if (typeof location === 'undefined') {
        CosmoScout.callNative('fly_to', planet);
      } else {
        CosmoScout.callNative('fly_to', planet, location.longitude, location.latitude, location.height, time);
      }

      CosmoScout.call('notifications', 'printNotification', 'Traveling', `to ${planet}`, 'send');
    }

    setCelestialBody(name) {
      CosmoScout.callNative('set_celestial_body', name);
    }
}
