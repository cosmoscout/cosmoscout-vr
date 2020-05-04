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
  _items;

  /**
   * @type {DataSet}
   */
  _itemsOverview;

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
  _currentSpeed = this._timeSpeedSteps.pause;

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

    this._items         = new vis.DataSet();
    this._itemsOverview = new vis.DataSet();

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
   * Adds a button to the button bar
   *
   * @param icon {string} Materialize icon name
   * @param tooltip {string} Tooltip text that gets shown if the button is hovered
   * @param callback {string} Name of callback on CosmoScout.callbacks
   */
  addButton(icon, tooltip, callback) {
    const button = CosmoScout.gui.loadTemplateContent('button');

    if (button === false) {
      return;
    }

    button.innerHTML = button.innerHTML.replace('%ICON%', icon).trim();

    button.setAttribute('title', tooltip);

    button.addEventListener('click', () => {
      CosmoScout.callbacks.find(callback)();
    });

    this._buttonContainer.appendChild(button);

    CosmoScout.gui.initTooltips();
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
    if (!(date instanceof Date)) {
      date = new Date(date);
    }

    if (isNaN(date.getTime())) {
      console.warning(`Failed to parse simulation time string: '${dateString}'!`);
    } else {
      this._centerTime = date;
      this._timeline.moveTo(this._centerTime, {
        animation: false,
      });
      this._timeline.setCustomTime(this._centerTime, this._timeId);
      this._updateOverviewLens();
      document.getElementById('date-label').innerText =
          CosmoScout.utils.formatDateReadable(this._centerTime);
    }
  }

  addEvent(id, name, description, start, end, color) {
    const data = {};
    data.start = new Date(start);
    data.id    = id;
    if (end !== '') {
      data.end = new Date(end);
    }
    data.name        = name;
    data.description = description;
    data.style       = "border-color: " + color;
    data.className   = `event event-${id}`;
    this._items.update(data);
    data.className = `overview-event event-${id}`;
    this._itemsOverview.update(data);
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
    CosmoScout.bookmarks.setVisible(true);
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

    const oldDate = new Date(this._centerTime.getTime());
    const newDate = new Date(this._centerTime.getTime());

    switch (type) {
    case 'year':
      newDate.setFullYear(newDate.getFullYear() + times);
      break;

    case 'month':
      newDate.setMonth(newDate.getMonth() + times);
      break;

    case 'day':
      newDate.setDate(newDate.getDate() + times);
      break;

    case 'hour':
      newDate.setHours(newDate.getHours() + times);
      break;

    case 'minute':
      newDate.setMinutes(newDate.getMinutes() + times);
      break;

    case 'second':
      newDate.setSeconds(newDate.getSeconds() + times);
      break;

    default:
      console.error('[data-type] not in [year, month, day, hour, second]');
      break;
    }

    const diff = newDate.getTime() - oldDate.getTime();

    this._centerTime.setSeconds(diff);

    const hoursDiff = diff / 1000 / 60 / 60;
    CosmoScout.callbacks.time.addHours(hoursDiff);
  }

  _initEventListener() {
    this._timelineContainer.addEventListener('wheel', this._manualZoomTimeline.bind(this), true);

    document.querySelectorAll('[data-change="time"]').forEach((element) => {
      if (element instanceof HTMLElement) {
        element.addEventListener('click', this._changeTime.bind(this));
        element.addEventListener('wheel', this._changeTime.bind(this));
      }
    });

    document.getElementById('pause-button').addEventListener('click', this._togglePause.bind(this));
    document.getElementById('speed-decrease-button')
        .addEventListener('click', this._decreaseSpeed.bind(this));
    document.getElementById('speed-increase-button')
        .addEventListener('click', this._increaseSpeed.bind(this));

    document.getElementById('time-reset-button')
        .addEventListener('click', this._resetTime.bind(this));

    document.getElementsByClassName('range-label')[0].addEventListener(
        'mousedown', () => this._setSpeed(this._timeSpeedSlider.noUiSlider.get()));

    document.getElementById('expand-button')
        .addEventListener('click', this._toggleOverview.bind(this));

    document.getElementById('calendar-button').addEventListener('click', () => {
      CosmoScout.calendar.setDate(this._timeline.getCustomTime(this._timeId));
      CosmoScout.calendar.toggle();
    });

    // toggle visibility of the increase / decrease time buttons -----------------------------------
    function mouseEnterTimeControl() {
      document.getElementById('increaseControl').classList.add('mouseNear');
      document.getElementById('decreaseControl').classList.add('mouseNear');
    }

    function mouseLeaveTimeControl() {
      document.getElementById('increaseControl').classList.remove('mouseNear');
      document.getElementById('decreaseControl').classList.remove('mouseNear');
    }

    function enterTimeButtons() {
      document.getElementById('increaseControl').classList.add('mouseNear');
      document.getElementById('decreaseControl').classList.add('mouseNear');
    }

    function leaveTimeButtons() {
      document.getElementById('increaseControl').classList.remove('mouseNear');
      document.getElementById('decreaseControl').classList.remove('mouseNear');
    }

    document.getElementById('time-control').onmouseenter = mouseEnterTimeControl;
    document.getElementById('time-control').onmouseleave = mouseLeaveTimeControl;

    document.getElementById('increaseControl').onmouseenter = enterTimeButtons;
    document.getElementById('increaseControl').onmouseleave = leaveTimeButtons;

    document.getElementById('decreaseControl').onmouseenter = enterTimeButtons;
    document.getElementById('decreaseControl').onmouseleave = leaveTimeButtons;
  }

  _toggleOverview() {
    this._overviewVisible = !this._overviewVisible;
    document.getElementById('timeline-container').classList.toggle('overview-visible');
    if (this._overviewVisible) {
      document.getElementById('expand-button').innerHTML =
          '<i class="material-icons">expand_less</i>';
    } else {
      document.getElementById('expand-button').innerHTML =
          '<i class="material-icons">expand_more</i>';
    }
  }

  /**
   * Rewinds the simulation and increases the speed if the simulation is already running backwards
   * @private
   */
  _decreaseSpeed() {
    if (this._timeSpeedSlider.noUiSlider.get() > 0) {
      this._timeSpeedSlider.noUiSlider.set(-1);
    } else if (this._currentSpeed === 0) {
      this._togglePause();
    } else {
      this._timeSpeedSlider.noUiSlider.set(this._currentSpeed - 1);
    }
  }

  /**
   * Increases the speed of the simulation
   * @private
   */
  _increaseSpeed() {
    if (this._timeSpeedSlider.noUiSlider.get() < 0) {
      this._timeSpeedSlider.noUiSlider.set(1);
    } else if (this._currentSpeed === 0) {
      this._togglePause();
    } else {
      this._timeSpeedSlider.noUiSlider.set(this._currentSpeed - (-1));
    }
  }

  /**
   * Resets the simulation time
   * @private
   */
  _resetTime() {
    this._overviewTimeline.setWindow(this._minDate, this._maxDate);
    this._timeSpeedSlider.noUiSlider.set(1);
    CosmoScout.callbacks.time.reset(3.0);
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

    this._timeline = new vis.Timeline(this._timelineContainer, this._items, this._timelineOptions);
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
        new vis.Timeline(overviewContainer, this._itemsOverview, this._overviewTimelineOptions);
    this._overviewTimeline.addCustomTime(this._timeline.getWindow().end, this._rightTimeId);
    this._overviewTimeline.addCustomTime(this._timeline.getWindow().start, this._leftTimeId);
    this._overviewTimeline.on('mouseUp', this._onMouseUp.bind(this));
    this._overviewTimeline.on('mouseDown', () => this._dragDistance = 0);
    this._overviewTimeline.on(
        'mouseMove', (e) => this._dragDistance += Math.abs(e.event.movementX));
    this._overviewTimeline.on('itemover', this._itemOverOverviewCallback.bind(this));
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
   * Shows a tooltip if an item is hovered
   *
   * @param properties
   * @private
   */
  _itemOverOverviewCallback(properties) {
    this._itemOverCallback(properties, true);
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
  _itemOverCallback(properties, overview) {
    document.getElementById('timeline-bookmark-tooltip-container').classList.add('visible');

    let eventData = this._items._data[properties.item];

    document.getElementById('timeline-bookmark-tooltip-name').innerHTML = eventData.name;
    document.getElementById('timeline-bookmark-tooltip-description').innerHTML =
        eventData.description;

    let eventDiv;
    if (overview) {
      eventDiv = document.querySelector(".vis-foreground .overview-event.event-" + eventData.id);
    } else {
      eventDiv = document.querySelector(".vis-foreground .event.event-" + eventData.id);
    }

    const eventRect    = eventDiv.getBoundingClientRect();
    const tooltipWidth = 300;
    const arrowWidth   = 10;
    const center       = eventRect.left + eventRect.width / 2;
    const left =
        Math.max(0, Math.min(document.body.offsetWidth - tooltipWidth, center - tooltipWidth / 2));
    document.getElementById('timeline-bookmark-tooltip-container').style.top =
        `${eventRect.bottom + arrowWidth}px`;
    document.getElementById('timeline-bookmark-tooltip-container').style.left = `${left}px`;
    document.getElementById('timeline-bookmark-tooltip-arrow').style.left =
        `${center - left - arrowWidth / 2}px`;
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

      CosmoScout.callbacks.time.setDate(CosmoScout.utils.formatDateCosmo(this._centerTime));
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
        let eventData = this._items._data[properties.item];
        CosmoScout.callbacks.time.setDate(CosmoScout.utils.formatDateCosmo(eventData.start), 3.0);
      } else if (properties.time != null) {
        CosmoScout.callbacks.time.setDate(
            CosmoScout.utils.formatDateCosmo(new Date(properties.time.getTime())), 3.0);
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
