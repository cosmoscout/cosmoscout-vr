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
    selectable: false,
    showCurrentTime: false,
    showTooltips: false,
    showWeekScale: true,
    orientation: {item: "top"},
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
    selectable: false,
    showCurrentTime: false,
    showWeekScale: true,
    showTooltips: false,
    zoomFriction: 3,
    orientation: {item: "top"},
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
  _zoomPercentage      = 0.004;
  _minRangeFactor      = 5;
  _maxRangeFactor      = 100000000;

  _overviewVisible = false;

  init() {
    this._buttonContainer   = document.getElementById('plugin-buttons');
    this._timelineContainer = document.getElementById('timeline');

    this._initTimeSpeedSlider();

    this._bookmarks         = new vis.DataSet();
    this._bookmarksOverview = new vis.DataSet();

    this._initTimelines();
    this._moveWindow();
    this._initEventListener();
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
      $(button).tooltip("dispose");
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

  addBookmark(id, start, end, color) {
    let data   = {};
    data.start = new Date(start);
    if (end !== '') {
      data.end = new Date(end);
    }
    data.id    = id;
    data.style = "border-color: " + color;
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
    // Zoom main timeline.
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
      CosmoScout.calendar.setDate(this._centerTime);
      CosmoScout.calendar.toggle();
    };

    // Search functionality.
    document.getElementById('timeline-search-button').onclick = ()   => this._executeSearch();
    document.querySelector('#timeline-search-area input').onkeypress = (e) => {
      if (e.keyCode == 13) {
        // Return pressed - try to travel to the location!
        this._executeSearch();
      }
    };
  }

  _executeSearch() {
    let query      = document.querySelector('#timeline-search-area input').value;
    let components = query.split(':');
    let planet     = CosmoScout.state.activePlanetCenter;

    if (components.length > 1) {
      if (components[0] != "") {
        planet = components[0];
      }

      if (components[1] != "") {
        query = components[1];
      } else {
        // The user entered only a body but no query. Fly to the body!
        CosmoScout.callbacks.navigation.setBody(planet, 5.0);
        return;
      }
    }

    CosmoScout.geocode.forward(planet, query, (location) => {
      if (location) {
        CosmoScout.callbacks.navigation.setBodyLongLatHeightDuration(
            planet, location.longitude, location.latitude, location.diameter * 1000, 5.0);
        CosmoScout.notifications.print("Travelling", "to " + location.name, "send");
      } else {
        CosmoScout.notifications.print("Not found", "No location matched the query", "error");
      }
    });
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
    this._overviewTimeline.on('rangechange', this._overviewDragCallback.bind(this));
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
   * Closes the tooltip if the mouse leaves the item and tooltip.
   * @private
   */
  _itemOutCallback() {
    CosmoScout.callbacks.bookmark.hideTooltip();
  }

  /**
   * Moves the displayed time window and sizes the time range according to the zoom factor.
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
    this._updateOverviewLens();
    this._timeline.setWindow(startDate, endDate, {
      animation: false,
    });
  }

  /**
   * Shows a tooltip if an item is hovered.
   *
   * @param properties {VisTimelineEvent}
   * @private
   */
  _itemOverCallback(properties) {
    let bookmark    = this._bookmarks.get(properties.item);
    const eventRect = properties.event.target.getBoundingClientRect();

    CosmoScout.callbacks.bookmark.showTooltip(
        bookmark.id, eventRect.left + eventRect.width / 2, eventRect.top + eventRect.height / 2);
  }

  /**
   * Called when the user moves the overview timeline.
   * @param properties {VisTimelineEvent}
   * @private
   */
  _overviewDragCallback(properties) {
    if (properties.byUser) {
      this._updateOverviewLens();
    }
  }

  /**
   * Sets the custom times on the overview that represent the left and right time on the timeline.
   * This clamps the start and end date of the overview lens so that they are not moved outside of
   * the screen to much.
   * @private
   */
  _updateOverviewLens() {
    let overviewWindow = this._overviewTimeline.getWindow();
    let overviewRange  = overviewWindow.end.getTime() - overviewWindow.start.getTime();

    let endTime = this._timeline.getWindow().end.getTime();
    endTime     = Math.max(endTime, overviewWindow.start.getTime() - overviewRange / 2);

    let startTime = this._timeline.getWindow().start.getTime();
    startTime     = Math.min(startTime, overviewWindow.end.getTime() + overviewRange / 2);

    this._overviewTimeline.setCustomTime(new Date(endTime), this._rightTimeId);
    this._overviewTimeline.setCustomTime(new Date(startTime), this._leftTimeId);
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
        let bookmark = this._bookmarks.get(properties.item);
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
