/* global IApi, CosmoScout, vis, CP, $, DateOperations, noUiSlider, Format */

/* eslint-disable class-methods-use-this, max-len, max-classes-per-file, no-underscore-dangle */

/**
 * https://visjs.github.io/vis-timeline/docs/timeline/#getEventProperties
 */
class VisTimelineEvent {
  /**
   * @type {boolean}
   */
  byUser;

  /**
   * @type {Number|null}
   */
  group;

  /**
   * @type {Number|null}
   */
  item;

  /**
   * @type {Number|null}
   */
  customTime;

  /**
   * @type {Number}
   */
  pageX;

  /**
   * @type {Number}
   */
  pageY;

  /**
   * @type {Number}
   */
  x;

  /**
   * @type {Number}
   */
  y;

  /**
   * @type {Date}
   */
  time;


  /**
   * @type {Date}
   */
  snappedTime;

  /**
   * @type {string|null}
   */
  what;

  /**
   * @type {Event}
   */
  event;


  /* Select Event */
  /**
   * @type {Number[]}
   */
  items;
}

/**
 * Timeline Api
 */
class TimelineApi extends IApi {
  name = 'timeline';

  PAUSE = 0;

  REALTIME = 1;

  MINUTES = 60;

  HOURS = 3600;

  DAYS = 86400;

  MONTHS = 2628000;

  _buttonContainer;

  /**
   * @type {DataSet}
   */
  _items;

  /**
   * @type {DataSet}
   */
  _itemsOverview;

  _pauseOptions = {
    moveable: true,
    zoomable: true,
  };

  _playingOptions = {
    moveable: false,
    zoomable: false,
  };

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
    moveable: false,
    showCurrentTime: false,
    editable: {
      add: true, // add new items by double tapping
      updateTime: true, // drag items horizontally
      updateGroup: false, // drag items from one group to another
      remove: false, // delete an item by tapping the delete button top right
      overrideItems: false, // allow these options to override item.editable
    },
    onAdd: this._onAddCallback.bind(this),
    onUpdate: this._onUpdateCallback.bind(this),
    onMove: this._onItemMoveCallback.bind(this),
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
      add: true, // add new items by double tapping
      updateTime: false, // drag items horizontally
      updateGroup: false, // drag items from one group to another
      remove: false, // delete an item by tapping the delete button top right
      overrideItems: false, // allow these options to override item.editable
    },
    onAdd: this._overviewOnAddCallback.bind(this),
    onUpdate: this._overviewOnUpdateCallback.bind(this),
    onMove: this._onItemMoveCallback.bind(this),
  };

  /**
   * @type {HTMLElement}
   * @member {noUiSlider}
   */
  _timeSpeedSlider;

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

  _click = false;

  _currentSpeed;

  /**
   * @type {Date}
   */
  _centerTime;

  _mouseOnTimelineDown = false;

  _mouseDownLeftTime;

  _minWidth = 30;

  _borderWidth = 3;

  _parHolder = {};

  _timelineContainer;

  _editingDoneOptions = {
    editable: {
      add: true, // add new items by double tapping
      updateTime: false, // drag items horizontally
      updateGroup: false, // drag items from one group to another
      remove: false, // delete an item by tapping the delete button top right
      overrideItems: false, // allow these options to override item.editable
    },
  };

  _whileEditingOptions = {
    editable: false,
  };

  _firstSliderValue = true;

  _rightTimeId = 'rightTime';

  _leftTimeId = 'leftTime';

  _timeId = 'custom';

  _tooltipVisible = false;

  _timelineRangeFactor = 100000;

  _hoveredHTMLEvent;

  _hoveredItem;

  _lastPlayValue = 1;

  _timelineZoomBlocked = true;

  _zoomPercentage = 0.2;

  _minRangeFactor = 5;

  _maxRangeFactor = 100000000;

  _wrongInputStyle = '2px solid red';

  _overviewVisible = false;

  _calenderVisible = false;

  _newCenterTimeId = 0;

  _newStartDateId = 1;

  _newEndDateId = 2;

  _state;

  init() {
    this._buttonContainer = document.getElementById('plugin-buttons');

    this._timelineContainer = document.getElementById('timeline');

    this._initTimeSpeedSlider();

    this._items = new vis.DataSet();
    this._itemsOverview = new vis.DataSet();

    this._initTimelines();
    this._moveWindow();
    this._initEventListener();
    this._initColorPicker();
    this._initCalendar();
  }

  /**
   * Adds a button to the button bar
   *
   * @param icon {string} Materialize icon name
   * @param tooltip {string} Tooltip text that gets shown if the button is hovered
   * @param callback {string} Function name passed to call_native
   */
  addButton(icon, tooltip, callback) {
    const button = CosmoScout.loadTemplateContent('button');

    if (button === false) {
      return;
    }

    button.innerHTML = button.innerHTML
      .replace('%ICON%', icon)
      .trim();

    button.setAttribute('title', tooltip);

    button.addEventListener('click', () => {
      CosmoScout.callNative(callback);
    });

    this._buttonContainer.appendChild(button);

    CosmoScout.initTooltips();
  }

  /**
   * Rotates the button bar compass
   *
   * @param angle {number}
   */
  setNorthDirection(angle) {
    document.getElementById('compass-arrow').style.transform = `rotateZ(${angle}rad)`;
  }

  setDate(date) {
    this._centerTime = new Date(date);
    this._timeline.moveTo(this._centerTime, {
      animation: false,
    });
    this._timeline.setCustomTime(this._centerTime, this._timeId);
    this._setOverviewTimes();
    document.getElementById('dateLabel').innerText = DateOperations.formatDateReadable(this._centerTime);
  }

  addItem(start, end, id, content, style, description, planet, place) {
    const data = {};
    data.start = new Date(start);
    data.id = id;
    if (end !== '') {
      data.end = new Date(end);
    }
    if (style !== '') {
      data.style = style;
    }
    data.planet = planet;
    data.description = description;
    data.place = place;
    data.content = content;
    data.className = `event ${id}`;
    this._items.update(data);
    data.className = `overviewEvent ${id}`;
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
   * Called from Application.cpp 622/816
   *
   * @param speed {number}
   */
  setTimeSpeed(speed) {
    let notification = [];

    switch (speed) {
      case this.PAUSE:
        this._setPause();
        notification = ['Pause', 'Time is paused.', 'pause'];
        break;

      case this.REALTIME:
        notification = ['Speed: Realtime', 'Time runs in realtime.', 'play_arrow'];
        break;

      case this.MINUTES:
        notification = ['Speed: Min/s', 'Time runs at one minute per second.', 'fast_forward'];
        break;

      case this.HOURS:
        notification = ['Speed: Hour/s', 'Time runs at one hour per second.', 'fast_forward'];
        break;

      case this.DAYS:
        notification = ['Speed: Day/s', 'Time runs at one day per second.', 'fast_forward'];
        break;

      case this.MONTHS:
        notification = ['Speed: Month/s', 'Time runs at one month per second.', 'fast_forward'];
        break;

      /* Negative times */
      case -this.REALTIME:
        notification = ['Speed: -Realtime', 'Time runs backwards in realtime.', 'fast_rewind'];
        break;

      case -this.MINUTES:
        notification = ['Speed: -Min/s', 'Time runs backwards at one minute per second.', 'fast_rewind'];
        break;

      case -this.HOURS:
        notification = ['Speed: -Hour/s', 'Time runs backwards at one hour per second.', 'fast_rewind'];
        break;

      case -this.DAYS:
        notification = ['Speed: -Day/s', 'Time runs backwards at one day per second.', 'fast_rewind'];
        break;

      case -this.MONTHS:
        notification = ['Speed: -Month/s', 'Time runs backwards at one month per second.', 'fast_rewind'];
        break;

      default:
        break;
    }

    if (notification.length > 0) {
      CosmoScout.notifications.printNotification(...notification);
    }
  }

  /**
   * Extracts the needed information out of the human readable place string
   * and calls fly_to_location for the given location.
   *
   * @param direct {boolean}
   * @param planet {string}
   * @param place {string} Location string in the form of '3.635° E 26.133° S 10.0 Tsd km'
   * @param name {string}
   */
  travelTo(direct, planet, place, name) {
    const placeArr = place.split(' ');

    const animationTime = direct ? 0 : 5;
    const location = {
      longitude: this._parseLongitude(placeArr[0], placeArr[1]),
      latitude: this._parseLatitude(placeArr[2], placeArr[3]),
      height: this._parseHeight(placeArr[4], placeArr[5]),
      name,
    };

    CosmoScout.callNative('fly_to_location', planet, location.longitude, location.latitude, location.height, animationTime);
    CosmoScout.notifications.printNotification('Travelling', `to ${location.name}`, 'send');
  }

  /* Internal methods */

  _initColorPicker() {
    const picker = new CP(document.querySelector('input[type="colorPicker"]'));

    picker.on('change', function change(color) {
      this.source.value = `#${color}`;
    });

    picker.on('change', (color) => {
      const colorField = document.getElementById('event-dialog-color');
      colorField.style.background = `#${color}`;
    });
  }

  /**
   * TODO remove jQuery
   * @private
   */
  _initCalendar() {
    $('#calendar')
      .datepicker({
        weekStart: 1,
        todayHighlight: true,
        maxViewMode: 3,
        format: 'yyyy-mm-dd',
        startDate: '1950-01-02',
        endDate: '2049-12-31',
      })
      .on('changeDate', this._changeDateCallback.bind(this));
  }

  /**
   * Snap back items if they were dragged with the mouse
   *
   * @param item
   * @param callback
   * @private
   */
  _onItemMoveCallback(item, callback) {
    callback(null);
  }

  _overviewOnUpdateCallback(item, callback) {
    this._onUpdateCallback(item, callback, true);
  }

  _overviewOnAddCallback(item, callback) {
    this._onAddCallback(item, callback, true);
  }

  /**
   * Called when an item is about to be added
   *
   * @param item
   * @param callback
   * @param overview
   * @private
   */
  _onAddCallback(item, callback, overview) {
    document.getElementById('event-dialog-name').style.border = '';
    document.getElementById('event-dialog-start-date').style.border = '';
    document.getElementById('event-dialog-description').style.border = '';
    this._timeline.setOptions(this._whileEditingOptions);
    this._overviewTimeline.setOptions(this._whileEditingOptions);
    document.getElementById('headlineForm').innerText = 'Add Event';
    document.getElementById('event-dialog-name').value = '';
    document.getElementById('add-event-dialog').style.display = 'block';
    document.getElementById('event-dialog-start-date').value = DateOperations.getFormattedDateWithTime(item.start);
    document.getElementById('event-dialog-end-date').value = '';
    document.getElementById('event-dialog-description').value = '';
    document.getElementById('event-dialog-planet').value = CosmoScout.statusbar.getActivePlanetCenter();

    let userPos = CosmoScout.statusbar.getObserverPosition();
    document.getElementById('event-dialog-location').value = Format.longitude(userPos[1]) + Format.latitude(userPos[0]) + Format.height(userPos[2]);
    this._parHolder.item = item;
    this._parHolder.callback = callback;
    this._parHolder.overview = overview;
    this._setPause();
  }

  /**
   * Called when an item is about to be updated
   * @param item
   * @param callback
   * @param overview
   * @private
   */
  _onUpdateCallback(item, callback, overview) {
    document.getElementById('event-dialog-name').style.border = '';
    document.getElementById('event-dialog-start-date').style.border = '';
    document.getElementById('event-dialog-description').style.border = '';
    this._timeline.setOptions(this._whileEditingOptions);
    this._overviewTimeline.setOptions(this._whileEditingOptions);
    document.getElementById('headlineForm').innerText = 'Update';
    document.getElementById('add-event-dialog').style.display = 'block';
    document.getElementById('event-dialog-name').value = item.content;
    document.getElementById('event-dialog-start-date').value = DateOperations.getFormattedDateWithTime(item.start);
    document.getElementById('event-dialog-description').value = item.description;
    document.getElementById('event-dialog-planet').value = item.planet;
    document.getElementById('event-dialog-location').value = item.place;
    if (item.end) {
      document.getElementById('event-dialog-end-date').value = DateOperations.getFormattedDateWithTime(item.end);
    } else {
      document.getElementById('event-dialog-end-date').value = '';
    }
    this._parHolder.item = item;
    this._parHolder.callback = callback;
    this._parHolder.overview = overview;
    this._setPause();
  }

  _setPause() {
    this._currentSpeed = 0;
    CosmoScout.callNative('set_time_speed', 0);
    document.getElementById('pause-button').innerHTML = '<i class="material-icons">play_arrow</i>';
    document.getElementsByClassName('range-label')[0].innerHTML = '<i class="material-icons">pause</i>';
    this._timeline.setOptions(this._pauseOptions);
    this._timelineZoomBlocked = false;
  }

  /**
   *
   * @param event {MouseEvent|WheelEvent}
   * @private
   */
  _changeTime(event) {
    const { type, direction } = event.target.dataset;

    if (typeof type === 'undefined' || typeof direction === 'undefined') {
      console.error('changeTime event bound to element without "data-type" and "data-direction" attributes.');
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
    CosmoScout.callNative('add_hours_without_animation', hoursDiff);
  }

  /**
   *
   * @param event {WheelEvent}
   * @private
   */
  _changeTimeScroll(event) {
    if (typeof event.target.dataset.diff === 'undefined') {
      return;
    }

    let diff = parseInt(event.target.dataset.diff, 10);
    // Data attribute is set in seconds. Call native wants hours
    diff = Math.abs(diff) / 3600;

    if (event.deltaY > 0) {
      diff = -diff;
    }

    CosmoScout.callNative('add_hours_without_animation', diff);
  }

  _initEventListener() {
    this._timelineContainer.addEventListener('wheel', this._manualZoomTimeline.bind(this), true);

    document.querySelectorAll('[data-change="time"]')
      .forEach((element) => {
        if (element instanceof HTMLElement) {
          element.addEventListener('click', this._changeTime.bind(this));
          element.addEventListener('wheel', this._changeTime.bind(this));
        }
      });

    document.getElementById('pause-button')
      .addEventListener('click', this._togglePause.bind(this));
    document.getElementById('speed-decrease-button')
      .addEventListener('click', this._decreaseSpeed.bind(this));
    document.getElementById('speed-increase-button')
      .addEventListener('click', this._increaseSpeed.bind(this));

    document.getElementById('event-tooltip-location')
      .addEventListener('click', this._travelToItemLocation.bind(this));

    document.getElementById('time-reset-button')
      .addEventListener('click', this._resetTime.bind(this));

    document.getElementsByClassName('range-label')[0].addEventListener('mousedown', this._rangeUpdateCallback.bind(this));


    document.getElementById('event-dialog-cancel-button')
      .addEventListener('click', this._closeForm.bind(this));
    document.getElementById('event-dialog-apply-button')
      .addEventListener('click', this._applyEvent.bind(this));


    document.getElementById('event-tooltip-container')
      .addEventListener('mouseleave', this._leaveCustomTooltip.bind(this));

    document.getElementById('expand-button')
      .addEventListener('click', this._toggleOverview.bind(this));


    document.getElementById('calendar-button')
      .addEventListener('click', this._enterNewCenterTime.bind(this));
    document.getElementById('dateLabel')
      .addEventListener('click', this._enterNewCenterTime.bind(this));

    // toggle visibility of the increase / decrease time buttons ---------------------------------------
    function mouseEnterTimeControl() {
      document.getElementById('increaseControl')
        .classList
        .add('mouseNear');
      document.getElementById('decreaseControl')
        .classList
        .add('mouseNear');
    }

    function mouseLeaveTimeControl() {
      document.getElementById('increaseControl')
        .classList
        .remove('mouseNear');
      document.getElementById('decreaseControl')
        .classList
        .remove('mouseNear');
    }

    function enterTimeButtons() {
      document.getElementById('increaseControl')
        .classList
        .add('mouseNear');
      document.getElementById('decreaseControl')
        .classList
        .add('mouseNear');
    }

    function leaveTimeButtons() {
      document.getElementById('increaseControl')
        .classList
        .remove('mouseNear');
      document.getElementById('decreaseControl')
        .classList
        .remove('mouseNear');
    }

    document.getElementById('time-control').onmouseenter = mouseEnterTimeControl;
    document.getElementById('time-control').onmouseleave = mouseLeaveTimeControl;

    document.getElementById('increaseControl').onmouseenter = enterTimeButtons;
    document.getElementById('increaseControl').onmouseleave = leaveTimeButtons;

    document.getElementById('decreaseControl').onmouseenter = enterTimeButtons;
    document.getElementById('decreaseControl').onmouseleave = leaveTimeButtons;
  }

  /**
   * Flies the observer to the location of the hovered item
   * @private
   */
  _travelToItemLocation() {
    this.travelTo(false, this._hoveredItem.planet, this._hoveredItem.place, this._hoveredItem.content);
  }

  /**
   * Close the event form
   * @private
   */
  _closeForm() {
    this._parHolder.callback(null); // cancel item creation
    document.getElementById('add-event-dialog').style.display = 'none';
    this._timeline.setOptions(this._editingDoneOptions);
    this._overviewTimeline.setOptions(this._editingDoneOptions);
  }

  _applyEvent() {
    /* TODO Just add a class to the parent element to indicate wrong state */
    if (document.getElementById('event-dialog-name').value !== ''
      && document.getElementById('event-dialog-start-date').value !== ''
      && document.getElementById('event-dialog-description').value !== '') {
      document.getElementById('event-dialog-name').style.border = '';
      document.getElementById('event-dialog-start-date').style.border = '';
      document.getElementById('event-dialog-description').style.border = '';
      this._parHolder.item.style = `border-color: ${document.getElementById('event-dialog-color').value}`;
      this._parHolder.item.content = document.getElementById('event-dialog-name').value;
      this._parHolder.item.start = new Date(document.getElementById('event-dialog-start-date').value);
      this._parHolder.item.description = document.getElementById('event-dialog-description').value;
      if (document.getElementById('event-dialog-end-date').value !== '') {
        this._parHolder.item.end = new Date(document.getElementById('event-dialog-end-date').value);
        const diff = this._parHolder.item.start - this._parHolder.item.end;
        if (diff >= 0) {
          this._parHolder.item.end = null;
          document.getElementById('event-dialog-end-date').style.border = this._wrongInputStyle;
          return;
        }
        document.getElementById('event-dialog-end-date').style.border = '';
      }
      this._parHolder.item.planet = document.getElementById('event-dialog-planet').value;
      this._parHolder.item.place = document.getElementById('event-dialog-location').value;
      if (this._parHolder.item.id == null) {
        this._parHolder.item.id = this._parHolder.item.content + this._parHolder.item.start + this._parHolder.item.end;
        this._parHolder.item.id = this._parHolder.item.id.replace(/\s/g, '');
      }
      if (this._parHolder.overview) {
        this._parHolder.item.className = `overviewEvent ${this._parHolder.item.id}`;
      } else {
        this._parHolder.item.className = `event ${this._parHolder.item.id}`;
      }
      this._parHolder.callback(this._parHolder.item); // send back adjusted new item
      document.getElementById('add-event-dialog').style.display = 'none';
      this._timeline.setOptions(this._editingDoneOptions);
      this._overviewTimeline.setOptions(this._editingDoneOptions);
      if (this._parHolder.overview) {
        this._parHolder.item.className = `event ${this._parHolder.item.id}`;
        this._items.update(this._parHolder.item);
      } else {
        this._parHolder.item.className = `overviewEvent ${this._parHolder.item.id}`;
        this._itemsOverview.update(this._parHolder.item);
      }
    } else {
      if (document.getElementById('event-dialog-name').value === '') {
        document.getElementById('event-dialog-name').style.border = this._wrongInputStyle;
      } else {
        document.getElementById('event-dialog-name').style.border = '';
      }
      if (document.getElementById('event-dialog-start-date').value === '') {
        document.getElementById('event-dialog-start-date').style.border = this._wrongInputStyle;
      } else {
        document.getElementById('event-dialog-start-date').style.border = '';
      }
      if (document.getElementById('event-dialog-description').value === '') {
        document.getElementById('event-dialog-description').style.border = this._wrongInputStyle;
      } else {
        document.getElementById('event-dialog-description').style.border = '';
      }
    }
  }

  _toggleOverview() {
    this._overviewVisible = !this._overviewVisible;
    document.getElementById('timeline-container')
      .classList
      .toggle('overview-visible');
    if (this._overviewVisible) {
      document.getElementById('expand-button').innerHTML = '<i class="material-icons">expand_less</i>';
    } else {
      document.getElementById('expand-button').innerHTML = '<i class="material-icons">expand_more</i>';
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
    CosmoScout.callNative('reset_time');
  }

  _togglePause() {
    if (this._currentSpeed !== 0) {
      this._setPause();
    } else {
      if (this._lastPlayValue === 0) {
        this._lastPlayValue = 1;
      }
      this._rangeUpdateCallback();
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

      this._timeSpeedSlider.noUiSlider.on('update', this._rangeUpdateCallback.bind(this));
    } catch (e) {
      console.error('Slider was already initialized');
    }
  }

  /**
   * Called at an interaction with the slider
   *
   * @private
   */
  _rangeUpdateCallback() {
    this._currentSpeed = this._timeSpeedSlider.noUiSlider.get();
    if (this._firstSliderValue) {
      document.getElementsByClassName('range-label')[0].innerHTML = '<i class="material-icons">chevron_right</i>';
      this._firstSliderValue = false;
      return;
    }

    document.getElementById('pause-button').innerHTML = '<i class="material-icons">pause</i>';
    this._timeline.setOptions(this._playingOptions);
    this._timelineZoomBlocked = true;
    if (parseInt(this._currentSpeed, 10) < 0) {
      document.getElementsByClassName('range-label')[0].innerHTML = '<i class="material-icons">chevron_left</i>';
    } else {
      document.getElementsByClassName('range-label')[0].innerHTML = '<i class="material-icons">chevron_right</i>';
    }

    this._moveWindow();

    switch (parseInt(this._currentSpeed, 10)) {
      case this._timeSpeedSteps.monthBack:
        CosmoScout.callNative('set_time_speed', -this.MONTHS);
        break;
      case this._timeSpeedSteps.dayBack:
        CosmoScout.callNative('set_time_speed', -this.DAYS);
        break;
      case this._timeSpeedSteps.hourBack:
        CosmoScout.callNative('set_time_speed', -this.HOURS);
        break;
      case this._timeSpeedSteps.minBack:
        CosmoScout.callNative('set_time_speed', -this.MINUTES);
        break;
      case this._timeSpeedSteps.secBack:
        CosmoScout.callNative('set_time_speed', -1);
        break;
      case this._timeSpeedSteps.secForward:
        CosmoScout.callNative('set_time_speed', 1);
        break;
      case this._timeSpeedSteps.minForward:
        CosmoScout.callNative('set_time_speed', this.MINUTES);
        break;
      case this._timeSpeedSteps.hourForward:
        CosmoScout.callNative('set_time_speed', this.HOURS);
        break;
      case this._timeSpeedSteps.dayForward:
        CosmoScout.callNative('set_time_speed', this.DAYS);
        break;
      case this._timeSpeedSteps.monthForward:
        CosmoScout.callNative('set_time_speed', this.MONTHS);
        break;
      default:
    }
  }

  /**
   * Creates the timelines and adds needed event listeners
   * @private
   */
  _initTimelines() {
    this._timelineContainer.addEventListener('wheel', this._manualZoomTimeline.bind(this), true);

    const overviewContainer = document.getElementById('overview');

    this._timeline = new vis.Timeline(this._timelineContainer, this._items, this._timelineOptions);
    this._centerTime = this._timeline.getCurrentTime();
    this._timeline.on('select', this._onSelect.bind(this));
    this._timeline.moveTo(this._centerTime, {
      animation: false,
    });

    this._timeline.addCustomTime(this._centerTime, this._timeId);
    this._timeline.on('click', this._onClickCallback.bind(this));
    this._timeline.on('changed', this._timelineChangedCallback.bind(this));
    this._timeline.on('mouseDown', this._mouseDownCallback.bind(this));
    this._timeline.on('mouseUp', this._mouseUpCallback.bind(this));
    this._timeline.on('rangechange', this._rangeChangeCallback.bind(this));
    this._timeline.on('itemover', this._itemOverCallback.bind(this));
    this._timeline.on('itemout', this._itemOutCallback.bind(this));

    // create overview timeline
    this._overviewTimeline = new vis.Timeline(overviewContainer, this._itemsOverview, this._overviewTimelineOptions);
    this._overviewTimeline.addCustomTime(this._timeline.getWindow().end, this._rightTimeId);
    this._overviewTimeline.addCustomTime(this._timeline.getWindow().start, this._leftTimeId);
    this._overviewTimeline.on('select', this._onSelect.bind(this));
    this._overviewTimeline.on('click', this._onClickCallback.bind(this));
    this._overviewTimeline.on('changed', this._drawFocusLens.bind(this));
    this._overviewTimeline.on('mouseDown', this._overviewMouseDownCallback.bind(this));
    this._overviewTimeline.on('rangechange', this._overviewChangedCallback.bind(this));
    this._overviewTimeline.on('itemover', this._itemOverOverviewCallback.bind(this));
    this._overviewTimeline.on('itemout', this._itemOutCallback.bind(this));
    this._initialOverviewWindow(new Date(1950, 1), new Date(2030, 12));
  }

  _initialOverviewWindow(start, end) {
    this._overviewTimeline.setWindow(start, end, {
      animations: false,
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

    if (element !== null && element.className !== 'event-tooltip') {
      document.getElementById('event-tooltip-container').style.display = 'none';
      this._tooltipVisible = false;
      this._hoveredHTMLEvent.classList.remove('mouseOver');
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
    const step = DateOperations.convertSeconds(this._timelineRangeFactor);
    let startDate = new Date(this._centerTime.getTime());
    let endDate = new Date(this._centerTime.getTime());
    startDate = DateOperations.decreaseDate(startDate, step.days, step.hours, step.minutes, step.seconds, step.milliSec);
    endDate = DateOperations.increaseDate(endDate, step.days, step.hours, step.minutes, step.seconds, step.milliSec);
    this._timeline.setWindow(startDate, endDate, {
      animations: false,
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
    document.getElementById('event-tooltip-container').style.display = 'block';
    this._tooltipVisible = true;
    for (const item in this._items._data) {
      if (this._items._data[item].id === properties.item) {
        document.getElementById('event-tooltip-content').innerHTML = this._items._data[item].content;
        document.getElementById('event-tooltip-description').innerHTML = this._items._data[item].description;
        document.getElementById('event-tooltip-location').innerHTML = `<i class='material-icons'>send</i> ${this._items._data[item].planet} ${this._items._data[item].place}`;
        this._hoveredItem = this._items._data[item];
      }
    }
    const events = document.getElementsByClassName(properties.item);
    let event;
    for (let i = 0; i < events.length; ++i) {
      if (!overview && $(events[i])
        .hasClass('event')) {
        event = events[i];
      } else if (overview && $(events[i])
        .hasClass('overviewEvent')) {
        event = events[i];
      }
    }
    this._hoveredHTMLEvent = event;
    this._hoveredHTMLEvent.classList.add('mouseOver');
    const eventRect = event.getBoundingClientRect();
    const left = eventRect.left - 150 < 0 ? 0 : eventRect.left - 150;
    document.getElementById('event-tooltip-container').style.top = `${eventRect.bottom}px`;
    document.getElementById('event-tooltip-container').style.left = `${left}px`;
  }

  /**
   * Hide the tooltip if the mouse leaves the tooltip
   * @private
   */
  _leaveCustomTooltip() {
    document.getElementById('event-tooltip-container').style.display = 'none';
    this._tooltipVisible = false;
    this._hoveredHTMLEvent.classList.remove('mouseOver');
  }

  /**
   * Sets variable values when a mouseDown event is triggered over the timeline
   * @private
   */
  _mouseDownCallback() {
    this._timeline.setOptions(this._pauseOptions);
    this._mouseOnTimelineDown = true;
    this._lastPlayValue = this._currentSpeed;
    this._click = true;
    this._mouseDownLeftTime = this._timeline.getWindow().start;
  }

  /**
   * Sets variable values when a mouseUp event is triggered over the timeline
   * @private
   */
  _mouseUpCallback() {
    if (this._mouseOnTimelineDown && this._lastPlayValue !== 0) {
      this._timeSpeedSlider.noUiSlider.set(parseInt(this._lastPlayValue, 10));
    }
    this._mouseOnTimelineDown = false;
  }

  /**
   * Callbacks to differ between a Click on the overview timeline and the user dragging the overview timeline
   * @private
   */
  _overviewMouseDownCallback() {
    this._click = true;
  }

  /**
   * Redraws the timerange indicator on the overview timeline in case the displayed time on the overview timeline changed
   * @private
   */
  _overviewChangedCallback() {
    this._click = false;
  }

  /**
   * Sets the custom times on the overview that represent the left and right time on the timeline
   * @private
   */
  _setOverviewTimes() {
    this._overviewTimeline.setCustomTime(this._timeline.getWindow().end, this._rightTimeId);
    this._overviewTimeline.setCustomTime(this._timeline.getWindow().start, this._leftTimeId);
    this._drawFocusLens();
  }

  /**
   * Redraws the timerange indicator on the overview timeline in case the displayed time on the timeline changed
   * @private
   */
  _timelineChangedCallback() {
    this._setOverviewTimes();
    this._drawFocusLens();
  }

  /**
   * Called when the user moves the timeline. It changes time so that the current time is alway in the middle
   * @param properties {VisTimelineEvent}
   * @private
   */
  _rangeChangeCallback(properties) {
    if (properties.byUser && String(properties.event) !== '[object WheelEvent]') {
      if (this._currentSpeed !== 0) {
        this._setPause();
      }
      this._click = false;
      const dif = properties.start.getTime() - this._mouseDownLeftTime.getTime();
      const secondsDif = dif / 1000;
      const hoursDif = secondsDif / 60 / 60;

      const step = DateOperations.convertSeconds(secondsDif);
      let date = new Date(this._centerTime.getTime());
      date = DateOperations.increaseDate(date, step.days, step.hours, step.minutes, step.seconds, step.milliSec);
      this._setDateLocal(date);
      this._mouseDownLeftTime = new Date(properties.start.getTime());
      CosmoScout.callNative('add_hours_without_animation', hoursDif);
    }
  }

  /**
   * Changes the shown date to a given date without synchronizing with CosmoScout VR
   * @param date {string} Date string
   * @private
   */
  _setDateLocal(date) {
    this._centerTime = new Date(date);
    this._timeline.moveTo(this._centerTime, {
      animation: false,
    });
    this._timeline.setCustomTime(this._centerTime, this._timeId);
    this._setOverviewTimes();
    document.getElementById('dateLabel').innerText = DateOperations.formatDateReadable(this._centerTime);
  }

  _drawFocusLens() {
    const leftCustomTime = document.getElementsByClassName('leftTime')[0];
    const leftRect = leftCustomTime.getBoundingClientRect();
    const rightCustomTime = document.getElementsByClassName('rightTime')[0];
    const rightRect = rightCustomTime.getBoundingClientRect();

    let divElement = document.getElementById('focus-lens');
    divElement.style.left = `${leftRect.right}px`;

    const height = leftRect.bottom - leftRect.top - 2;
    let width = rightRect.right - leftRect.left;

    let xValue = 0;
    if (width < this._minWidth) {
      width = this._minWidth + 2 * this._borderWidth;
      xValue = -(leftRect.left + this._minWidth - rightRect.right) / 2 - this._borderWidth;
      xValue = Math.round(xValue);
      divElement.style.transform = ` translate(${xValue}px, 0px)`;
    } else {
      divElement.style.transform = ' translate(0px, 0px)';
    }

    divElement.style.height = `${height}px`;
    divElement.style.width = `${width}px`;

    divElement = document.getElementById('focus-lens-left');
    width = leftRect.right + xValue + this._borderWidth;
    width = width < 0 ? 0 : width;
    divElement.style.width = `${width}px`;
    const body = document.getElementsByTagName('body')[0];
    const bodyRect = body.getBoundingClientRect();

    divElement = document.getElementById('focus-lens-right');
    width = bodyRect.right - rightRect.right + xValue + 1;
    width = width < 0 ? 0 : width;
    divElement.style.width = `${width}px`;
  }

  /**
   * Called if the timeline is clicked
   * @param properties
   * @private
   */
  _onClickCallback(properties) {
    if (this._click) {
      this._generalOnClick(properties);
    }
  }

  /**
   * Changes the size of the displayed time range while the simulation is still playing
   *
   * @param event
   * @private
   */
  _manualZoomTimeline(event) {
    if (this._timelineZoomBlocked) {
      if (event.deltaY < 0) {
        this._timelineRangeFactor -= this._timelineRangeFactor * this._zoomPercentage;
        if (this._timelineRangeFactor < this._minRangeFactor) {
          this._timelineRangeFactor = this._minRangeFactor;
        }
      } else {
        this._timelineRangeFactor += this._timelineRangeFactor * this._zoomPercentage;
        if (this._timelineRangeFactor > this._maxRangeFactor) {
          this._timelineRangeFactor = this._maxRangeFactor;
        }
      }
      this._rangeUpdateCallback();
    }
  }

  /**
   * Vis Timeline Event Properties
   * https://visjs.github.io/vis-timeline/docs/timeline/#getEventProperties
   * TODO .id compared to array
   *
   * @param properties {VisTimelineEvent}
   * @private
   */
  _onSelect(properties) {
    for (const item in this._items._data) {
      if (this._items._data[item].id === properties.items) {
        const dif = this._items._data[item].start.getTime() - this._centerTime.getTime();
        let hoursDif = dif / 1000 / 60 / 60;

        if (this._items._data[item].start.getTimezoneOffset() > this._centerTime.getTimezoneOffset()) {
          hoursDif -= 1;
        } else if (this._items._data[item].start.getTimezoneOffset() < this._centerTime.getTimezoneOffset()) {
          hoursDif += 1;
        }

        CosmoScout.callNative('add_hours', hoursDif);
        this.travelTo(true, this._items._data[item].planet, this._items._data[item].place, this._items._data[item].content);
      }
    }
  }

  /**
   * Vis Timeline Event Properties
   * https://visjs.github.io/vis-timeline/docs/timeline/#getEventProperties
   *
   * @param properties {VisTimelineEvent}
   * @private
   */
  _generalOnClick(properties) {
    if (properties.what !== 'item' && properties.time != null) {
      const dif = properties.time.getTime() - this._centerTime.getTime();
      let hoursDif = dif / 1000 / 60 / 60;

      if (properties.time.getTimezoneOffset() > this._centerTime.getTimezoneOffset()) {
        hoursDif -= 1;
      } else if (properties.time.getTimezoneOffset() < this._centerTime.getTimezoneOffset()) {
        hoursDif += 1;
      }
      CosmoScout.callNative('add_hours', hoursDif);
    }
  }

  /**
   * Parses a latitude string for travelTo
   *
   * @see {travelTo}
   * @param lat {string}
   * @param half {string}
   * @return {number}
   * @private
   */
  _parseLatitude(lat, half) {
    let latitude = parseFloat(lat);
    if (half === 'S') {
      latitude = -latitude;
    }

    return latitude;
  }

  /**
   * Parses a longitude string for travelTo
   *
   * @see {travelTo}
   * @param lon {string}
   * @param half {string}
   * @return {number}
   * @private
   */
  _parseLongitude(lon, half) {
    let longitude = parseFloat(lon);
    if (half === 'W') {
      longitude = -longitude;
    }

    return longitude;
  }

  /**
   * Parses a height string
   *
   * @param heightStr {string}
   * @param unit {string}
   * @return {number}
   * @private
   */
  _parseHeight(heightStr, unit) {
    const height = parseFloat(heightStr);

    switch (unit) {
      case 'mm':
        return height / 1000;
      case 'cm':
        return height / 100;
      case 'm':
        return height;
      case 'km':
        return height * 1e3;
      case 'Tsd':
        return height * 1e6;
      case 'AU':
        return height * 1.496e11;
      case 'ly':
        return height * 9.461e15;
      case 'pc':
        return height * 3.086e16;
      default:
        return height * 3.086e19;
    }
  }

  /**
   * Sets the visibility of the calendar to the given value(true or false)
   * @param visible {boolean}
   * @private
   */
  _setVisible(visible) {
    if (visible) {
      $('#calendar')
        .addClass('visible');
    } else {
      $('#calendar')
        .removeClass('visible');
    }
  }

  /**
   * Toggles the calendar visibility
   * @private
   */
  _toggleVisible() {
    if (this._calenderVisible) {
      this._calenderVisible = false;
      this._setVisible(false);
    } else {
      this._calenderVisible = true;
      this._setVisible(true);
    }
  }

  /**
   * Called if the Calendar is used to change the date
   * @private
   */
  _enterNewCenterTime() {
    $('#calendar')
      .datepicker('update', this._timeline.getCustomTime(this._timeId));
    if (this._calenderVisible && this._state === this._newCenterTimeId) {
      this._toggleVisible();
    } else if (!this._calenderVisible) {
      this._state = this._newCenterTimeId;
      this._toggleVisible();
    }
  }

  /*
  // Called if the Calendar is used to enter a start date of an event
  function enter_start_date() {
      if (state === newStartDateId) {
          toggle_visible();
      } else {
          state = newStartDateId;
          calenderVisible = true;
          set_visible(true);
      }
  }


  // Called if the Calendar is used to enter the end date of an event
  function enter_end_date() {
      if (state === newEndDateId) {
          toggle_visible();
      } else {
          state = newEndDateId;
          calenderVisible = true;
          set_visible(true);
      }
  } */

  /**
   * Sets the time to a specific date
   * @param date {Date}
   * @private
   */
  _setTimeToDate(date) {
    date.setHours(12);
    CosmoScout.callNative('set_date', DateOperations.formatDateCosmo(new Date(date.getTime())));
    const startDate = new Date(date.getTime());
    const endDate = new Date(date.getTime());
    startDate.setHours(0);
    endDate.setHours(24);
    this._setPause();
    this._timeline.setWindow(startDate, endDate, {
      animation: false,
    });
    this._setOverviewTimes();
  }

  /**
   * Called if an Date in the Calendar is picked
   * @param event
   * @private
   */
  _changeDateCallback(event) {
    this._toggleVisible();
    switch (this._state) {
      case this._newCenterTimeId:
        this._setTimeToDate(event.date);
        break;
      case this._newStartDateId:
        document.getElementById('event-dialog-start-date').value = event.format();
        break;
      case this._newEndDateId:
        document.getElementById('event-dialog-end-date').value = event.format();
        break;
      default:
      // code block
    }
  }
}
