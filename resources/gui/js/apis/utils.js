/* eslint no-param-reassign: 0 */

/**
 * This is a default CosmoScout API. Once initialized, you can access its methods via
 * CosmoScout.utils.<method name>.
 */
class UtilsApi extends IApi {
  /**
   * @inheritDoc
   */
  name = 'utils';

  _defaultLocale = 'de-de';

  _defaultDateOptions = {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  };

  /**
   * Format a Date to a for a human readable string DD.MM.YYYY HH:MM:SS.
   *
   * @param date {Date}
   * @return {string}
   */
  formatDateReadable(date) {
    return `${date.toLocaleDateString(this._defaultLocale, this._defaultDateOptions)} ${date.toLocaleTimeString(this._defaultLocale)}`;
  }

  /**
   * Format a Date to YYYY-MM-DD
   *
   * @param date {Date}
   * @return {string}
   */
  getFormattedDate(date) {
    return date.toISOString().split('T')[0];
  }

  /**
   * Format a Date to YYYY-MM-DD HH:MM:SS.
   *
   * @param date {Date}
   * @return {string}
   */
  getFormattedDateWithTime(date) {
    return `${this.getFormattedDate(date)} ${date.toLocaleTimeString('de-de')}`;
  }

  /**
   * Format a Date to a readable format for CosmoScout VR YYYY-MM-DD HH:MM:SS.sss.
   *
   * @param date {Date}
   * @return {string}
   */
  formatDateCosmo(date) {
    const milli = date.getMilliseconds().toString().padStart(3, '0');

    return `${this.getFormattedDateWithTime(date)}.${milli}`;
  }

  /**
   * Convert seconds into an object containing the duration in hours -- ms
   *
   * @param seconds {number}
   * @return {{}}
   */
  convertSeconds(seconds) {
    const mSec = 60;
    const hSec = mSec * mSec;
    const dSec = hSec * 24;

    const converted = {};

    converted.days = Math.floor(seconds / dSec);

    const daysSec = converted.days * dSec;
    converted.hours = Math.floor((seconds - daysSec) / hSec);

    const hoursSec = converted.hours * hSec;
    converted.minutes = Math.floor((seconds - daysSec - hoursSec) / mSec);

    const minSec = converted.minutes * mSec;
    converted.seconds = Math.floor(seconds - daysSec - hoursSec - minSec);

    converted.milliSec = Math.round((seconds - Math.floor(seconds)) * 1000);

    return converted;
  }

  /**
   * Increase a Date by days, hours, minutes, seconds and milliseconds.
   *
   * @param date {Date}
   * @param days {number}
   * @param hours {number}
   * @param minutes {number}
   * @param seconds {number}
   * @param milliSec {number}
   * @return {Date}
   */
  increaseDate(date, days, hours, minutes, seconds, milliSec) {
    date.setDate(date.getDate() + days);
    date.setHours(date.getHours() + hours);
    date.setMinutes(date.getMinutes() + minutes);
    date.setSeconds(date.getSeconds() + seconds);
    date.setMilliseconds(date.getMilliseconds() + milliSec);
    return date;
  }

  /**
   * Decrease a Date by days, hours, minutes, seconds and milliseconds.
   *
   * @param date {Date}
   * @param days {number}
   * @param hours {number}
   * @param minutes {number}
   * @param seconds {number}
   * @param milliSec {number}
   * @return {Date}
   */
  decreaseDate(date, days, hours, minutes, seconds, milliSec) {
    return this.increaseDate(date, -days, -hours, -minutes, -seconds, -milliSec);
  }


  /**
   * @param number {number|string}
   * @return {string}
   */
  formatNumber(number) {
    number = Number(number);

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
   * Returns a formatted height string.
   *
   * @param height {number|string}
   * @return {string}
   */
  formatHeight(height) {
    let num;
    let unit;

    height = Number(height);

    if (Math.abs(height) < 0.1) {
      num = this.formatNumber(height * 1000);
      unit = 'mm';
    } else if (Math.abs(height) < 1) {
      num = this.formatNumber(height * 100);
      unit = 'cm';
    } else if (Math.abs(height) < 1e4) {
      num = this.formatNumber(height);
      unit = 'm';
    } else if (Math.abs(height) < 1e7) {
      num = this.formatNumber(height / 1e3);
      unit = 'km';
    } else if (Math.abs(height) < 1e10) {
      num = this.formatNumber(height / 1e6);
      unit = 'Tsd km';
    } else if (Math.abs(height / 1.496e11) < 1e4) {
      num = this.formatNumber(height / 1.496e11);
      unit = 'AU';
    } else if (Math.abs(height / 9.461e15) < 1e3) {
      num = this.formatNumber(height / 9.461e15);
      unit = 'ly';
    } else if (Math.abs(height / 3.086e16) < 1e3) {
      num = this.formatNumber(height / 3.086e16);
      unit = 'pc';
    } else {
      num = this.formatNumber(height / 3.086e19);
      unit = 'kpc';
    }

    return `${num} ${unit}`;
  }

  /**
   * Returns a formatted speed string.
   *
   * @param speed {number|string}
   * @return {string}
   */
  formatSpeed(speed) {
    let num;
    let unit;

    speed = Number(speed);

    if (Math.abs(speed * 3.6) < 500) {
      num = this.formatNumber(speed * 3.6);
      unit = 'km/h';
    } else if (Math.abs(speed) < 1e3) {
      num = this.formatNumber(speed);
      unit = 'm/s';
    } else if (Math.abs(speed) < 1e7) {
      num = this.formatNumber(speed / 1e3);
      unit = 'km/s';
    } else if (Math.abs(speed) < 1e8) {
      num = this.formatNumber(speed / 1e6);
      unit = 'Tsd km/s';
    } else if (Math.abs(speed / 2.998e8) < 1e3) {
      num = this.formatNumber(speed / 2.998e8);
      unit = 'SoL';
    } else if (Math.abs(speed / 1.496e11) < 1e3) {
      num = this.formatNumber(speed / 1.496e11);
      unit = 'AU/s';
    } else if (Math.abs(speed / 9.461e15) < 1e3) {
      num = this.formatNumber(speed / 9.461e15);
      unit = 'ly/s';
    } else if (Math.abs(speed / 3.086e16) < 1e3) {
      num = this.formatNumber(speed / 3.086e16);
      unit = 'pc/s';
    } else {
      num = this.formatNumber(speed / 3.086e19);
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
  formatLatitude(lat) {
    lat = Number(lat);

    if (lat < 0) {
      return `${(-lat).toFixed(2)}° S `;
    }

    return `${(lat).toFixed(2)}° N `;
  }

  /**
   * Returns a formatted longitude string.
   *
   * @param lon {number|string}
   * @return {string}
   */
  formatLongitude(lon) {
    lon = Number(lon);

    if (lon < 0) {
      return `${(-lon).toFixed(2)}° W `;
    }

    return `${(lon).toFixed(2)}° E `;
  }

  /**
   * @param number {number}
   * @return {string|number}
   */
  beautifyNumber(number) {
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
