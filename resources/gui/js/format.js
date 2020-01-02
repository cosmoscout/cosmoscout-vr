/* eslint no-param-reassign: 0 */

/**
 * Formats different numbers
 */
// eslint-disable-next-line no-unused-vars
class Format {
  /**
     * @param number {number|string}
     * @return {string}
     */
  static number(number) {
    number = parseFloat(number);

    // Set very small numbers to 0
    if (number < Number.EPSILON && -Number.EPSILON > number) {
      number = 0;
    }

    if (Math.abs(number) < 10) {
      return number.toFixed(2);
    } if (Math.abs(number) < 100) {
      return number.toFixed(1);
    }

    return number.toFixed(0);
  }

  /**
     * Returns a formatted height string
     *
     * @param height {number|string}
     * @return {string}
     */
  static height(height) {
    let num;
    let unit;

    height = parseFloat(height);

    if (Math.abs(height) < 0.1) {
      num = Format.number(height * 1000);
      unit = 'mm';
    } else if (Math.abs(height) < 1) {
      num = Format.number(height * 100);
      unit = 'cm';
    } else if (Math.abs(height) < 1e4) {
      num = Format.number(height);
      unit = 'm';
    } else if (Math.abs(height) < 1e7) {
      num = Format.number(height / 1e3);
      unit = 'km';
    } else if (Math.abs(height) < 1e10) {
      num = Format.number(height / 1e6);
      unit = 'Tsd km';
    } else if (Math.abs(height / 1.496e11) < 1e4) {
      num = Format.number(height / 1.496e11);
      unit = 'AU';
    } else if (Math.abs(height / 9.461e15) < 1e3) {
      num = Format.number(height / 9.461e15);
      unit = 'ly';
    } else if (Math.abs(height / 3.086e16) < 1e3) {
      num = Format.number(height / 3.086e16);
      unit = 'pc';
    } else {
      num = Format.number(height / 3.086e19);
      unit = 'kpc';
    }

    return `${num} ${unit}`;
  }

  /**
     * Returns a formatted speed string
     *
     * @param speed {number|string}
     * @return {string}
     */
  static speed(speed) {
    let num;
    let unit;

    speed = parseFloat(speed);

    if (Math.abs(speed * 3.6) < 500) {
      num = Format.number(speed * 3.6);
      unit = 'km/h';
    } else if (Math.abs(speed) < 1e3) {
      num = Format.number(speed);
      unit = 'm/s';
    } else if (Math.abs(speed) < 1e7) {
      num = Format.number(speed / 1e3);
      unit = 'km/s';
    } else if (Math.abs(speed) < 1e8) {
      num = Format.number(speed / 1e6);
      unit = 'Tsd km/s';
    } else if (Math.abs(speed / 2.998e8) < 1e3) {
      num = Format.number(speed / 2.998e8);
      unit = 'SoL';
    } else if (Math.abs(speed / 1.496e11) < 1e3) {
      num = Format.number(speed / 1.496e11);
      unit = 'AU/s';
    } else if (Math.abs(speed / 9.461e15) < 1e3) {
      num = Format.number(speed / 9.461e15);
      unit = 'ly/s';
    } else if (Math.abs(speed / 3.086e16) < 1e3) {
      num = Format.number(speed / 3.086e16);
      unit = 'pc/s';
    } else {
      num = Format.number(speed / 3.086e19);
      unit = 'kpc/s';
    }

    return `${num} ${unit}`;
  }

  /**
     * Returns a formatted latitude string
     *
     * @param lat {number|string}
     * @return {string}
     */
  static latitude(lat) {
    lat = parseFloat(lat);

    if (lat < 0) {
      return `${(-lat).toFixed(2)}° S `;
    }

    return `${(lat).toFixed(2)}° N `;
  }

  /**
     * Returns a formatted longitude string
     *
     * @param lon {number|string}
     * @return {string}
     */
  static longitude(lon) {
    lon = parseFloat(lon);

    if (lon < 0) {
      return `${(-lon).toFixed(2)}° W `;
    }

    return `${(lon).toFixed(2)}° E `;
  }

  /**
     *
     * @param number {number}
     * @return {string|number}
     */
  static beautifyNumber(number) {
    const abs = Math.abs(number);
    let value;

    if (abs >= 10000) {
      value = Number(number.toPrecision(2)).toExponential();
    } else if (abs >= 1000) {
      value = Number(number.toPrecision(4));
    } else if (abs >= 1) {
      value = Number(number.toPrecision(3));
    } else if (abs >= 0.1) {
      value = Number(number.toPrecision(2));
    } else if (abs === 0) {
      value = '0';
    } else {
      value = Number(number.toPrecision(2)).toExponential();
    }

    return value.toString();
  }
}
