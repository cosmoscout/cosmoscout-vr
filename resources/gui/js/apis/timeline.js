/* global IApi, CosmoScout, vis, CP, $, DateOperations, noUiSlider, Format */

/* eslint-disable class-methods-use-this, max-len, max-classes-per-file, no-underscore-dangle */

/**
 * Timeline Api
 */
class TimelineApi extends IApi {
  name = 'timeline';

  /**
   * Conversions to seconds.
   */
  PAUSE    = 0;
  REALTIME = 1;
  MINUTES  = 60;
  HOURS    = 3600;
  DAYS     = 86400;
  MONTHS   = 2628000;

  /**
   * @type {DataSet}
   */
  _bookmarks;

  /**
   * @type {DataSet}
   */
  _bookmarksOverview;

  /**
   * @type {Timeline}
   */
  _timeline;

  _timelineOptions = {
    minHeight: 35,
    maxHeight: 35,
    stack: false,
    max: new Date(2030, 12),
    min: new Date(1950, 1),
    moment:
        function(date) {
          return vis.moment(date).utc(); // Use UTC
        },
    zoomable: false,
    moveable: true,
    showCurrentTime: false,
    editable: {
      add: true,            // add new items by double tapping
      updateTime: false,    // drag items horizontally
      updateGroup: false,   // drag items from one group to another
      remove: false,        // delete an item by tapping the delete button top right
      overrideItems: false, // allow these options to override item.editable
    },
    onAdd: this._addBookmarkCallback.bind(this),
    format: {
      minorLabels: {
        millisecond: 'SSS[ms]',
        second: 's[s]',
        minute: 'HH:mm',
        hour: 'HH:mm',
        weekday: 'ddd D',
        day: 'ddd D',
        week: 'MMM D',
        month: 'MMM',
        year: 'YYYY',
      },
      majorLabels: {
        millisecond: 'HH:mm:ss',
        second: 'D MMMM HH:mm',
        minute: 'ddd D MMMM',
        hour: 'ddd D MMMM',
        weekday: 'MMMM YYYY',
        day: 'MMMM YYYY',
        week: 'MMMM YYYY',
        month: 'YYYY',
        year: '',
      },
    },
  };

  /**
   * @type {Timeline}
   */
  _overviewTimeline;

  _overviewTimelineOptions = {
    minHeight: 40,
    maxHeight: 40,
    stack: false,
    max: new Date(2030, 12),
    min: new Date(1950, 1),
    moment:
        function(date) {
          return vis.moment(date).utc(); // Use UTC
        },
    zoomMin: 100000, // Do not zoom to milliseconds on the overview timeline
    zoomable: true,
    moveable: true,
    showCurrentTime: false,
    editable: {
      add: true,            // add new items by double tapping
      updateTime: false,    // drag items horizontally
      updateGroup: false,   // drag items from one group to another
      remove: false,        // delete an item by tapping the delete button top right
      overrideItems: false, // allow these options to override item.editable
    },
    onAdd: this._addBookmarkCallback.bind(this)
  };

  /**
   * @type {HTMLElement}
   * @member {noUiSlider}
   */
  _timeSpeedSlider;
  _firstSliderValue = true;

  _timeSpeedSteps = {
    pause: 0,
    secForward: 1,
    minForward: 2,
    hourForward: 3,
    dayForward: 4,
    monthForward: 5,
    secBack: -1,
    minBack: -2,
    hourBack: -3,
    dayBack: -4,
    monthBack: -5,
  };

  /**
   * Stores one of the values above.
   */
  _currentSpeed = this._timeSpeedSteps.secForward;

  /**
   * Used to restore playback state after a timeline drag.
   */
  _beforeDragSpeed = this._currentSpeed;

  /**
   * Used to differentiate between a click and a drag.
   */
  _dragDistance = false;

  /**
   * @type {Date}
   */
  _centerTime;

  /**
   * Parameters configuring the overview lens.
   */
  _minWidth    = 30;
  _borderWidth = 3;

  _buttonContainer;
  _timelineContainer;

  /**
   *  IDs to locate specific time points on the timeline.
   */
  _rightTimeId = 'overview-lens-right-time';
  _leftTimeId  = 'overview-lens-left-time';
  _timeId      = 'center-time';

  /**
   * Zoom parameters.
   */
  _timelineRangeFactor = 100000;
  _zoomPercentage      = 0.002;
  _minRangeFactor      = 5;
  _maxRangeFactor      = 100000000;

  _overviewVisible = false;

  init() {
    this._buttonContainer = document.getElementById('plugin-buttons');

    this._timelineContainer = document.getElementById('timeline');

    this._initTimeSpeedSlider();

    this._bookmarks         = new vis.DataSet();
    this._bookmarksOverview = new vis.DataSet();

    this._initTimelines();
    this._moveWindow();
    this._initEventListener();
    this._updateOverviewLens();
  }

  /**
   * Called once a frame by CosmoScout VR.
   */
  update() {
    this.setDate(CosmoScout.state.simulationTime);
  }

  /**
   * Adds a button to the button bar.
   *
   * @param name {string} Tooltip text that gets shown if the button is hovered
   * @param icon {string} Materialize icon name
   * @param callback {string} Name of callback on CosmoScout.callbacks
   */
  addButton(name, icon, callback) {
    const button     = CosmoScout.gui.loadTemplateContent('button');
    button.innerHTML = button.innerHTML.replace('%ICON%', icon).trim();
    button.setAttribute('title', name);
    button.onclick = () => {
      CosmoScout.callbacks.find(callback)();
    };

    this._buttonContainer.appendChild(button);

    CosmoScout.gui.initTooltips();
  }

  /**
   * Removes a button from the button bar.
   *
   * @param name {string} Tooltip text that gets shown if the button is hovered
   */
  removeButton(name) {
    const button = this._buttonContainer.querySelector(`[data-original-title="${name}"]`);

    if (button) {
      button.remove();
    }
  }

  /**
   * Rotates the button bar compass
   *
   * @param angle {number}
   */
  setNorthDirection(angle) {
    document.getElementById('compass-arrow').style.transform = `rotateZ(${angle}rad)`;
  }

  /**
   * Sets the current date on the timeline
   *
   * @param date {Date or string}
   */
  setDate(date) {
    if (isNaN(date.getTime())) {
      console.warn("Invalid date given to timeline!");
    } else {
      this._centerTime = date;
      this._timeline.moveTo(this._centerTime, {
        animation: false,
      });
      this._timeline.setCustomTime(this._centerTime, this._timeId);
      this._updateOverviewLens();

      let dateText = this._centerTime.toISOString().replace('T', ' ').slice(0, 19);
      document.getElementById('date-label').innerText = dateText;
    }
  }

  addBookmark(id, name, description, start, end, color, hasLocation) {
    let data   = {};
    data.start = new Date(start);
    if (end !== '') {
      data.end = new Date(end);
    }
    data.id          = id;
    data.name        = name;
    data.description = description;
    data.style       = "border-color: " + color;
    data.hasLocation = hasLocation == true;
    this._bookmarks.update(data);
    this._bookmarksOverview.update(data);
  }

  removeBookmark(id) {
    this._bookmarks.remove(id);
    this._bookmarksOverview.remove(id);
  }

  /**
   * Sets the min and max date for the timeline
   *
   * @param min {string} Date string
   * @param max {string} Date string
   */
  setTimelineRange(min, max) {
    const rangeOpt = {
      min,
      max,
    };
    this._minDate = min;
    this._maxDate = max;
    this._timeline.setOptions(rangeOpt);
    this._overviewTimeline.setOptions(rangeOpt);
    this._initialOverviewWindow(new Date(min), new Date(max));
  }

  /**
   * Called when an item is about to be added
   * @private
   */
  _addBookmarkCallback() {
    CosmoScout.bookmarkEditor.addNewBookmark();
    this._setSpeed(0);
  }

  /**
   *
   * @param event {MouseEvent|WheelEvent}
   * @private
   */
  _changeTime(event) {
    const {type, direction} = event.target.dataset;

    if (typeof type === 'undefined' || typeof direction === 'undefined') {
      console.error(
          'changeTime event bound to element without "data-type" and "data-direction" attributes.');
      return;
    }

    let times = 1;
    if (typeof event.deltaY !== 'undefined') {
      times = -Math.sign(event.deltaY);
    } else if (direction === 'decrease') {
      times = -times;
    }

    const newDate = new Date(this._centerTime.getTime());

    switch (type) {
    case 'year':
      newDate.setUTCFullYear(newDate.getUTCFullYear() + times);
      break;
    case 'month':
      newDate.setUTCMonth(newDate.getUTCMonth() + times);
      break;
    case 'day':
      newDate.setUTCDate(newDate.getUTCDate() + times);
      break;
    case 'hour':
      newDate.setUTCHours(newDate.getUTCHours() + times);
      break;
    case 'minute':
      newDate.setUTCMinutes(newDate.getUTCMinutes() + times);
      break;
    case 'second':
      newDate.setUTCSeconds(newDate.getUTCSeconds() + times);
      break;
    default:
      console.error('[data-type] not in [year, month, day, hour, second]');
      break;
    }

    CosmoScout.callbacks.time.setDate(newDate.toISOString());
  }

  _initEventListener() {
    this._timelineContainer.addEventListener('wheel', this._manualZoomTimeline.bind(this), true);

    // Handlers for the year / month / day / hour / ... -up-and-down-buttons.
    document.querySelectorAll('[data-change="time"]').forEach((element) => {
      if (element instanceof HTMLElement) {
        element.addEventListener('click', this._changeTime.bind(this));
        element.addEventListener('wheel', this._changeTime.bind(this));
      }
    });

    // Toggle pause.
    document.getElementById('pause-button').onclick = () => this._togglePause();

    // Handler for speed-decrease button.
    document.getElementById('speed-decrease-button').onclick = () => {
      if (this._timeSpeedSlider.noUiSlider.get() > 0) {
        this._timeSpeedSlider.noUiSlider.set(-1);
      } else if (this._currentSpeed === 0) {
        this._togglePause();
      } else {
        this._timeSpeedSlider.noUiSlider.set(this._currentSpeed - 1);
      }
    };

    // Handler for speed-increase button.
    document.getElementById('speed-increase-button').onclick = () => {
      if (this._timeSpeedSlider.noUiSlider.get() < 0) {
        this._timeSpeedSlider.noUiSlider.set(1);
      } else if (this._currentSpeed === 0) {
        this._togglePause();
      } else {
        this._timeSpeedSlider.noUiSlider.set(this._currentSpeed - (-1));
      }
    };

    // Reset timeline state with the reset button.
    document.getElementById('time-reset-button').onclick = () => {
      this._overviewTimeline.setWindow(this._minDate, this._maxDate);
      this._timeSpeedSlider.noUiSlider.set(1);
      CosmoScout.callbacks.time.reset(3.0);
    };

    // Start the simulation time when clicking on the speed slider.
    document.getElementsByClassName('range-label')[0].addEventListener(
        'mousedown', () => this._setSpeed(this._timeSpeedSlider.noUiSlider.get()));

    // Toggle the overview with the tiny button on the right.
    document.getElementById('expand-button').onclick = () => {
      this._overviewVisible = !this._overviewVisible;
      document.getElementById('timeline-container').classList.toggle('overview-visible');
      if (this._overviewVisible) {
        document.getElementById('expand-button').innerHTML =
            '<i class="material-icons">expand_less</i>';
      } else {
        document.getElementById('expand-button').innerHTML =
            '<i class="material-icons">expand_more</i>';
      }
    };

    // Show calendar on calender button clicks.
    document.getElementById('calendar-button').onclick = () => {
      CosmoScout.calendar.setDate(this._timeline.getCustomTime(this._timeId));
      CosmoScout.calendar.toggle();
    };
  }

  _togglePause() {
    if (this._currentSpeed == 0) {
      this._setSpeed(this._timeSpeedSlider.noUiSlider.get());
    } else {
      this._setSpeed(0);
    }
  }

  _initTimeSpeedSlider() {
    this._timeSpeedSlider = document.getElementById('range');

    try {
      noUiSlider.create(this._timeSpeedSlider, {
        range: {
          min: this._timeSpeedSteps.monthBack,
          '4.5%': this._timeSpeedSteps.dayBack,
          '9%': this._timeSpeedSteps.hourBack,
          '13.5%': this._timeSpeedSteps.minBack,
          '18%': this._timeSpeedSteps.secBack,
          '82%': this._timeSpeedSteps.secForward,
          '86.5%': this._timeSpeedSteps.minForward,
          '91%': this._timeSpeedSteps.hourForward,
          '95.5%': this._timeSpeedSteps.dayForward,
          max: this._timeSpeedSteps.monthForward,
        },
        snap: true,
        start: 1,
      });

      this._timeSpeedSlider.noUiSlider.on('update', () => {
        let speed = this._timeSpeedSlider.noUiSlider.get();
        if (this._firstSliderValue) {
          document.getElementsByClassName('range-label')[0].innerHTML =
              '<i class="material-icons">chevron_right</i>';
          this._firstSliderValue = false;
          return;
        }
        this._setSpeed(speed);
      });
    } catch (e) { console.error('Slider was already initialized'); }
  }

  /**
   * Called at an interaction with the slider
   *
   * @private
   */
  _setSpeed(speed) {
    this._currentSpeed = parseInt(speed);

    if (this._currentSpeed == 0) {
      document.getElementById('pause-button').innerHTML =
          '<i class="material-icons">play_arrow</i>';
    } else {
      document.getElementById('pause-button').innerHTML = '<i class="material-icons">pause</i>';
    }

    if (this._currentSpeed < 0) {
      document.getElementsByClassName('range-label')[0].innerHTML =
          '<i class="material-icons">chevron_left</i>';
    } else if (this._currentSpeed > 0) {
      document.getElementsByClassName('range-label')[0].innerHTML =
          '<i class="material-icons">chevron_right</i>';
    } else {
      document.getElementsByClassName('range-label')[0].innerHTML =
          '<i class="material-icons">pause</i>';
    }

    switch (this._currentSpeed) {
    case this._timeSpeedSteps.monthBack:
      CosmoScout.callbacks.time.setSpeed(-this.MONTHS);
      break;
    case this._timeSpeedSteps.dayBack:
      CosmoScout.callbacks.time.setSpeed(-this.DAYS);
      break;
    case this._timeSpeedSteps.hourBack:
      CosmoScout.callbacks.time.setSpeed(-this.HOURS);
      break;
    case this._timeSpeedSteps.minBack:
      CosmoScout.callbacks.time.setSpeed(-this.MINUTES);
      break;
    case this._timeSpeedSteps.secBack:
      CosmoScout.callbacks.time.setSpeed(-1);
      break;
    case 0:
      CosmoScout.callbacks.time.setSpeed(0);
      break;
    case this._timeSpeedSteps.secForward:
      CosmoScout.callbacks.time.setSpeed(1);
      break;
    case this._timeSpeedSteps.minForward:
      CosmoScout.callbacks.time.setSpeed(this.MINUTES);
      break;
    case this._timeSpeedSteps.hourForward:
      CosmoScout.callbacks.time.setSpeed(this.HOURS);
      break;
    case this._timeSpeedSteps.dayForward:
      CosmoScout.callbacks.time.setSpeed(this.DAYS);
      break;
    case this._timeSpeedSteps.monthForward:
      CosmoScout.callbacks.time.setSpeed(this.MONTHS);
      break;
    default:
    }
  }

  /**
   * Creates the timelines and adds needed event listeners
   * @private
   */
  _initTimelines() {
    const overviewContainer = document.getElementById('overview');

    this._timeline =
        new vis.Timeline(this._timelineContainer, this._bookmarks, this._timelineOptions);
    this._centerTime = this._timeline.getCurrentTime();
    this._timeline.moveTo(this._centerTime, {
      animation: false,
    });

    this._timeline.addCustomTime(this._centerTime, this._timeId);
    this._timeline.on('mouseUp', this._onMouseUp.bind(this));
    this._timeline.on('mouseDown', () => this._dragDistance = 0);
    this._timeline.on('mouseMove', (e) => {this._dragDistance += Math.abs(e.event.movementX)});
    this._timeline.on('rangechange', this._timelineDragCallback.bind(this));
    this._timeline.on('rangechanged', this._timelineDragEndCallback.bind(this));
    this._timeline.on('itemover', this._itemOverCallback.bind(this));
    this._timeline.on('itemout', this._itemOutCallback.bind(this));

    // create overview timeline
    this._overviewTimeline =
        new vis.Timeline(overviewContainer, this._bookmarksOverview, this._overviewTimelineOptions);
    this._overviewTimeline.addCustomTime(this._timeline.getWindow().end, this._rightTimeId);
    this._overviewTimeline.addCustomTime(this._timeline.getWindow().start, this._leftTimeId);
    this._overviewTimeline.on('mouseUp', this._onMouseUp.bind(this));
    this._overviewTimeline.on('mouseDown', () => this._dragDistance = 0);
    this._overviewTimeline.on(
        'mouseMove', (e) => this._dragDistance += Math.abs(e.event.movementX));
    this._overviewTimeline.on('itemover', this._itemOverCallback.bind(this));
    this._overviewTimeline.on('itemout', this._itemOutCallback.bind(this));
    this._initialOverviewWindow(new Date(1950, 1), new Date(2030, 12));
  }

  _initialOverviewWindow(start, end) {
    this._overviewTimeline.setWindow(start, end, {
      animation: false,
    });
  }

  /**
   * Closes the tooltip if the mouse leaves the item and tooltip
   *
   * @param properties {VisTimelineEvent}
   * @private
   */
  _itemOutCallback(properties) {
    const element = properties.event.toElement;

    if (element !== null) {
      document.getElementById('timeline-bookmark-tooltip-container').classList.remove('visible');
    }
  }

  /**
   * Moves the displayed time window and sizes the time range according to the zoom factor
   * @private
   */
  _moveWindow() {
    const step    = CosmoScout.utils.convertSeconds(this._timelineRangeFactor);
    let startDate = new Date(this._centerTime.getTime());
    let endDate   = new Date(this._centerTime.getTime());
    startDate     = CosmoScout.utils.decreaseDate(
        startDate, step.days, step.hours, step.minutes, step.seconds, step.milliSec);
    endDate = CosmoScout.utils.increaseDate(
        endDate, step.days, step.hours, step.minutes, step.seconds, step.milliSec);
    this._timeline.setWindow(startDate, endDate, {
      animation: false,
    });
  }

  /**
   * TODO this iterates over the private _data field from DataSet
   * Shows a tooltip if an item is hovered
   *
   * @param properties {VisTimelineEvent}
   * @param overview {boolean} True if target is the upper timeline
   * @private
   */
  _itemOverCallback(properties) {
    document.getElementById('timeline-bookmark-tooltip-container').classList.add('visible');

    let bookmark = this._bookmarks._data[properties.item];

    if (bookmark.hasLocation) {
      document.getElementById('timeline-bookmark-tooltip-goto-location').classList.remove('hidden');
      document.getElementById('timeline-bookmark-tooltip-goto-location').onclick = () => {
        CosmoScout.callbacks.bookmark.gotoLocation(bookmark.id);
      };
    } else {
      document.getElementById('timeline-bookmark-tooltip-goto-location').classList.add('hidden');
    }

    document.getElementById('timeline-bookmark-tooltip-goto-time').onclick = () => {
      CosmoScout.callbacks.bookmark.gotoTime(bookmark.id, 2.0);
    };

    document.getElementById('timeline-bookmark-tooltip-edit').onclick = () => {
      CosmoScout.callbacks.bookmark.edit(bookmark.id);
    };

    document.getElementById('timeline-bookmark-tooltip-name').innerHTML = bookmark.name;
    document.getElementById('timeline-bookmark-tooltip-description').innerHTML =
        bookmark.description;

    const eventRect    = properties.event.target.getBoundingClientRect();
    const tooltipWidth = 400;
    const arrowWidth   = 10;
    const center       = eventRect.left + eventRect.width / 2;
    const left =
        Math.max(0, Math.min(document.body.offsetWidth - tooltipWidth, center - tooltipWidth / 2));
    document.getElementById('timeline-bookmark-tooltip-container').style.top =
        `${eventRect.bottom + arrowWidth + 5}px`;
    document.getElementById('timeline-bookmark-tooltip-container').style.left = `${left}px`;
    document.getElementById('timeline-bookmark-tooltip-arrow').style.left =
        `${center - left - arrowWidth}px`;
  }

  /**
   * Sets the custom times on the overview that represent the left and right time on the timeline
   * @private
   */
  _updateOverviewLens() {
    this._overviewTimeline.setCustomTime(this._timeline.getWindow().end, this._rightTimeId);
    this._overviewTimeline.setCustomTime(this._timeline.getWindow().start, this._leftTimeId);

    const leftCustomTime  = document.getElementsByClassName(this._leftTimeId)[0];
    const leftRect        = leftCustomTime.getBoundingClientRect();
    const rightCustomTime = document.getElementsByClassName(this._rightTimeId)[0];
    const rightRect       = rightCustomTime.getBoundingClientRect();

    let divElement        = document.getElementById('focus-lens');
    divElement.style.left = `${leftRect.right}px`;

    const height = leftRect.bottom - leftRect.top - 2;
    let width    = rightRect.right - leftRect.left;

    let xValue = 0;
    if (width < this._minWidth) {
      width  = this._minWidth + 2 * this._borderWidth;
      xValue = -(leftRect.left + this._minWidth - rightRect.right) / 2 - this._borderWidth;
      xValue = Math.round(xValue);
      divElement.style.transform = ` translate(${xValue}px, 0px)`;
    } else {
      divElement.style.transform = ' translate(0px, 0px)';
    }

    divElement.style.height = `${height}px`;
    divElement.style.width  = `${width}px`;

    divElement             = document.getElementById('focus-lens-left');
    width                  = leftRect.right + xValue + this._borderWidth;
    width                  = width < 0 ? 0 : width;
    divElement.style.width = `${width}px`;
    const body             = document.getElementsByTagName('body')[0];
    const bodyRect         = body.getBoundingClientRect();

    divElement             = document.getElementById('focus-lens-right');
    width                  = bodyRect.right - rightRect.right + xValue + 1;
    width                  = width < 0 ? 0 : width;
    divElement.style.width = `${width}px`;
  }

  /**
   * Called when the user moves the timeline. It changes time so that the current time is alway in
   * the middle
   * @param properties {VisTimelineEvent}
   * @private
   */
  _timelineDragCallback(properties) {
    if (properties.byUser) {
      if (this._currentSpeed !== 0) {
        this._beforeDragSpeed = this._currentSpeed;
        this._setSpeed(0);
      }

      this._centerTime = new Date(properties.start.getTime() / 2 + properties.end.getTime() / 2);
      this._timeline.setCustomTime(this._centerTime, this._timeId);
      this._updateOverviewLens();

      window.callNative("time.setDate", this._centerTime.toISOString());
    }
  }

  /**
   * Called when the user moved the timeline. It resets the playing state to before.
   *
   * @param properties {VisTimelineEvent}
   * @private
   */
  _timelineDragEndCallback(properties) {
    if (properties.byUser) {
      this._setSpeed(this._beforeDragSpeed);
      this._beforeDragSpeed = 0;
    }
  }

  /**
   * Called when the timeline is clicked.
   *
   * @param properties
   * @private
   */
  _onMouseUp(properties) {
    if (this._dragDistance < 10) {
      if (properties.item != null) {
        let bookmark = this._bookmarks._data[properties.item];
        CosmoScout.callbacks.time.setDate(bookmark.start.toISOString(), 3.0);
      } else if (properties.time != null) {
        CosmoScout.callbacks.time.setDate(new Date(properties.time.getTime()).toISOString(), 3.0);
      }
    }
  }

  /**
   * Changes the size of the displayed time range while the simulation is still playing.
   *
   * @param event
   * @private
   */
  _manualZoomTimeline(event) {
    this._timelineRangeFactor += this._timelineRangeFactor * this._zoomPercentage * event.deltaY;
    this._timelineRangeFactor =
        Math.max(this._minRangeFactor, Math.min(this._maxRangeFactor, this._timelineRangeFactor));
    this._moveWindow();
  }
}
