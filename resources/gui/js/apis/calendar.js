/* global IApi, CosmoScout */

/**
 * The Calendar API
 */
class CalendarApi extends IApi {
  /**
   * @inheritDoc
   */
  name = 'calendar';

  /**
   * @type {HTMLElement}
   */
  _calendarWindow;

  /**
   * @type {jQueryElement}
   */
  _calendar;

  /**
   * @inheritDoc
   */
  init() {
    this._calendarWindow = document.querySelector("#calendar");
    this._calendar       = $('#calendar .window-content')
                         .datepicker({
                           weekStart: 1,
                           todayHighlight: true,
                           maxViewMode: 3,
                           format: 'yyyy-mm-dd',
                           startDate: '1950-01-02',
                           endDate: '2049-12-31',
                         })
                         .on('changeDate', () => {
                           let date = $('#calendar .window-content').datepicker("getUTCDate");
                           date.setUTCHours(12);
                           CosmoScout.callbacks.time.setDate(date.toISOString());
                           if (!this._calendarWindow.locked) {
                             this.setVisible(false);
                           }
                         });
  }

  /**
   * Sets the visibility of the calendar to the given value (true or false).
   *
   * @param visible {boolean}
   */
  setVisible(visible) {
    if (visible) {
      this._calendarWindow.classList.add("visible");
    } else {
      this._calendarWindow.classList.remove("visible");
    }
  }

  /**
   * Toggles the visibility of the calendar.
   */
  toggle() {
    this.setVisible(!this._calendarWindow.classList.contains("visible"));
  }

  /**
   * Sets the current date shown in the calendar.
   *
   * @param date {Date}
   */
  setDate(date) {
    this._calendar.datepicker('update', date);
  }
}
