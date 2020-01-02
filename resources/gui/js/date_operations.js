/**
 * Locales won't work in Android WebView
 */
// eslint-disable-next-line no-unused-vars
class DateOperations {
  static _defaultLocale = 'de-de';

  static _defaultDateOptions = {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  };

  /**
   * Set a locale for all formatDateReadable calls
   *
   * @param {string} locale
   */
  static setLocale(locale) {
    try {
      new Date().toLocaleString('i');
    } catch (e) {
      console.error('Browser does not support setting date locales.');

      return;
    }

    this._defaultLocale = locale;
  }

  /**
   *Format a Date to a for a human readable string DD.MM.YYYY HH:MM:SS
   *
   * @param date {Date}
   * @return {string}
   */
  static formatDateReadable(date) {
    return `${date.toLocaleDateString(this._defaultLocale, this._defaultDateOptions)} ${date.toLocaleTimeString(this._defaultLocale)}`;
  }

  /**
   * Format a Date to YYYY-MM-DD
   *
   * @param date {Date}
   * @return {string}
   */
  static getFormattedDate(date) {
    return date.toISOString().split('T')[0];
  }

  /**
   * Format a Date to YYYY-MM-DD HH:MM:SS
   *
   * @param date {Date}
   * @return {string}
   */
  static getFormattedDateWithTime(date) {
    return `${this.getFormattedDate(date)} ${date.toLocaleTimeString('de-de')}`;
  }

  /**
   * Format a Date to a readable format for CosmoScoutVR YYYY-MM-DD HH:MM:SS.sss
   *
   * @param date {Date}
   * @return {string}
   */
  static formatDateCosmo(date) {
    const milli = date.getMilliseconds().toString().padStart(3, '0');

    return `${this.getFormattedDateWithTime(date)}.${milli}`;
  }

  /**
   * Convert seconds into an object containing the duration in hours -- ms
   *
   * @param seconds {number}
   * @return {{}}
   */
  static convertSeconds(seconds) {
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
   * Increase a Date by days, hours , minutes, seconds and milliseconds
   *
   * @param date {Date}
   * @param days {number}
   * @param hours {number}
   * @param minutes {number}
   * @param seconds {number}
   * @param milliSec {number}
   * @return {Date}
   */
  static increaseDate(date, days, hours, minutes, seconds, milliSec) {
    date.setDate(date.getDate() + days);
    date.setHours(date.getHours() + hours);
    date.setMinutes(date.getMinutes() + minutes);
    date.setSeconds(date.getSeconds() + seconds);
    date.setMilliseconds(date.getMilliseconds() + milliSec);
    return date;
  }

  /**
   * Decrease a Date by days, hours , minutes, seconds and milliseconds
   *
   * @param date {Date}
   * @param days {number}
   * @param hours {number}
   * @param minutes {number}
   * @param seconds {number}
   * @param milliSec {number}
   * @return {Date}
   */
  static decreaseDate(date, days, hours, minutes, seconds, milliSec) {
    return this.increaseDate(date, -days, -hours, -minutes, -seconds, -milliSec);
  }
}
