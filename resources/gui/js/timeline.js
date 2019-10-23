// API calls - these should be called from C++ -----------------------------------------------------

function init() {
    $("#timeline-container").addClass("visible");
}

// Sets the timeline to the given date
function set_date(date) { }

// Prints a notifivcatio for the time-speed and changes the slider if the time is paused
function set_time_speed(speed) { }

// Adds a new event to the timeline
function add_item(start, end, id, content, style, description, planet, place) { }

// Add a Button to the button bar
// @param icon The materialize icon to use
// @param tooltip Tooltip text that gets shown if the button is hovered
// @param callback Native function that gets called if the button is clicked. The function has
//                  to be registered as callback before clicking the button.
function add_button(icon, tooltip, callback) {
    var button = document.createElement("a");
    button.setAttribute('class', "btn light-glass");
    button.setAttribute('data-toggle', 'tooltip');
    button.setAttribute('title', tooltip);
    callback = "window.call_native('" + callback + "')";
    button.setAttribute("onClick", callback);
    var iconElement = document.createElement("i");
    iconElement.innerHTML = icon;
    iconElement.setAttribute("class", "material-icons");
    button.appendChild(iconElement);
    document.getElementById("plugin-buttons").appendChild(button);
    $('[data-toggle="tooltip"]').tooltip({ delay: 500, placement: "top", html: false });
}

function set_north_direction(angle) {
    $("#compass-arrow").css("transform", "rotateZ(" + angle + "rad)");
}

// Sets the active planet
function set_active_planet(center) { }

// Sets the position of the user
function set_user_position(long, lat, height) { }

// Sets the min and max date for the timeline
function set_timeline_range(min, max) { }

// timeline configuration --------------------------------------------------------------------------

let monthInSec = 2628000;
let dayInSec = 86400;
let hourInSec = 3600;
let minuteInSec = 60;

let secondInHours = 1.0 / (60.0 * 60.0);
let minuteInHours = 1.0 / 60.0;
let dayInHours = 24;

let timeId = 'custom';

let leftTimeId = 'leftTime';
let rightTimeId = 'rightTime';

var drawFocusLensCallback = null;

var firstTime = true;

let zoomPercentage = 0.2;
var timelineZoomBlocked = true;

var firstSliderValue = true;

let paus = 0;
let secForw = 1;
let hourForw = 2;
let dayForw = 3;
let monthForw = 4;
let secBack = -1;
let hourBack = -2;
let dayBack = -3;
let monthBack = -4;

var timelineRangeFactor = 100000;
let maxRangeFactor = 100000000;
let minRangeFactor = 5;

let redrawRate = 16.666666;
let secSpeed = 0.0166666;
let hourSpeed = 60;
let daySpeed = 1440;
let monthSpeed = 43800;

let startOfDay = 0;
let middleOfDay = 12;
let endOfDay = 24;

var currentSpeed;

var parHolder = new Object();

var timeline;
var overviewTimeLine;
var centerTime;

var click;
var mouseDownLeftTime;

var minDate;
var maxDate;

var activePlanetCenter;
var userPosition = new Object();

// DOM element where the Timeline will be attached
var timelineContainer = document.getElementById('timeline');
var overviewContainer = document.getElementById('overview');

// Configuration for the Timeline
var options = {
    minHeight: 35,
    maxHeight: 35,
    stack: false,
    max: new Date(2030, 12),
    min: new Date(1950, 1),
    zoomable: false,
    moveable: false,
    showCurrentTime: false,
    editable: {
        add: true,         // add new items by double tapping
        updateTime: true,  // drag items horizontally
        updateGroup: false, // drag items from one group to another
        remove: false,       // delete an item by tapping the delete button top right
        overrideItems: false  // allow these options to override item.editable
    },
    onAdd: on_add_callback,
    onUpdate: on_update_callback,
    onMove: on_item_move_callback,
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
            year: 'YYYY'
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
            year: ''
        }
    }
};

var playingOpt = {
    moveable: false,
    zoomable: false
}

var pausOpt = {
    moveable: true,
    zoomable: true
}

var overviewOptions = {
    minHeight: 40,
    maxHeight: 40,
    stack: false,
    max: new Date(2030, 12),
    min: new Date(1950, 1),
    zoomable: true,
    moveable: true,
    showCurrentTime: false,
    editable: {
        add: true,         // add new items by double tapping
        updateTime: false,  // drag items horizontally
        updateGroup: false, // drag items from one group to another
        remove: false,       // delete an item by tapping the delete button top right
        overrideItems: false  // allow these options to override item.editable
    },
    onAdd: overview_on_add_callback,
    onUpdate: overview_on_update_callback,
    onMove: on_item_move_overview_callback
}

var whileEditingOpt = {
    editable: false
}

var editingDoneOpt = {
    editable: {
        add: true,         // add new items by double tapping
        updateTime: false,  // drag items horizontally
        updateGroup: false, // drag items from one group to another
        remove: false,       // delete an item by tapping the delete button top right
        overrideItems: false  // allow these options to override item.editable
    }
}

var animationFalse = {
    animation: false
};

var lastPlayValue = secForw;
var mouseOnTimelineDown = false;

//create range
var range = document.getElementById('range');

noUiSlider.create(range, {
    range: {
        'min': monthBack,
        '6%': dayBack,
        '12%': hourBack,
        '18%': secBack,
        '82%': secForw,
        '88%': hourForw,
        '94%': dayForw,
        'max': monthForw,
    },
    snap: true,
    start: secForw
});

range.noUiSlider.on('update', range_update_callback);

var items = new vis.DataSet();
var itemsOverview = new vis.DataSet();
// Create a Timeline
timeline = new vis.Timeline(timelineContainer, items, options);
centerTime = timeline.getCurrentTime();
timeline.on('select', on_select);
timeline.moveTo(centerTime, animationFalse);
timeline.addCustomTime(centerTime, timeId);
timeline.on('click', on_click_callback);
timeline.on('changed', timeline_change_callback);
timeline.on('mouseDown', mouse_down_callback);
timeline.on('mouseUp', mouse_up_callback);
timeline.on('rangechange', range_change_callback);
timeline.on('itemover', item_over_callback);
timeline.on('itemout', item_out_callback);

//create overview timeline
overviewTimeLine = new vis.Timeline(overviewContainer, itemsOverview, overviewOptions);
overviewTimeLine.addCustomTime(timeline.getWindow().end, rightTimeId);
overviewTimeLine.addCustomTime(timeline.getWindow().start, leftTimeId);
overviewTimeLine.on('select', on_select);
overviewTimeLine.on('click', on_overview_click);
overviewTimeLine.on('changed', overview_change_callback);
overviewTimeLine.on('mouseDown', overviewMouse_down_callback);
overviewTimeLine.on('rangechange', overviewRange_change_callback);
overviewTimeLine.on('itemover', item_over_overview_callback);
overviewTimeLine.on('itemout', item_out_callback);
initial_overview_window(new Date(1950, 1), new Date(2030, 12));

document.getElementById("dateLabel").innerText = format_date_readable(centerTime);

move_window(secSpeed);

// Sets the active planet
function set_active_planet(center) {
    activePlanetCenter = center;
}


function format_number(number) {
    if (Math.abs(number) < 10) return number.toFixed(2);
    else if (Math.abs(number) < 100) return number.toFixed(1);
    else return number.toFixed(0)
}

function format_height(height) {
    if (Math.abs(height) < 0.1) return format_number(height * 1000) + ' mm';
    else if (Math.abs(height) < 1) return format_number(height * 100) + ' cm';
    else if (Math.abs(height) < 1e4) return format_number(height) + ' m';
    else if (Math.abs(height) < 1e7) return format_number(height / 1e3) + ' km';
    else if (Math.abs(height) < 1e10) return format_number(height / 1e6) + ' Tsd km';
    else if (Math.abs(height / 1.496e11) < 1e4) return format_number(height / 1.496e11) + ' AU';
    else if (Math.abs(height / 9.461e15) < 1e3) return format_number(height / 9.461e15) + ' ly';
    else if (Math.abs(height / 3.086e16) < 1e3) return format_number(height / 3.086e16) + ' pc';

    return format_number(height / 3.086e19) + ' kpc';
}

function format_latitude(lat) {
    if (lat < 0)
        return (-lat).toFixed(2) + "° S ";
    else
        return (lat).toFixed(2) + "° N ";
}

function format_longitude(long) {
    if (long < 0)
        return (-long).toFixed(2) + "° W ";
    else
        return (long).toFixed(2) + "° E ";
}

function set_user_position(long, lat, height) {
    userPosition.long = long;
    userPosition.lat = lat;
    userPosition.height = height;
}

// Redraws the tooltip of an event while the event is visible
function redraw_tooltip(event) {
    return new Promise(resolve => {
        var eventRect = event.getBoundingClientRect();
        var left = eventRect.left - 150 < 0 ? 0 : eventRect.left - 150;
        document.getElementById("event-tooltip-container").style.top = eventRect.bottom + 'px';
        document.getElementById("event-tooltip-container").style.left = left + 'px';

        setTimeout(function () {
            resolve(10);
            if (tooltipVisible) {
                redraw_tooltip(event);
            }
        }, redrawRate);
    });
}

// Starts redrawing the tooltip of an event
async function start_redraw_tooltip(event) {
    await redraw_tooltip(event);
}

var hoveredItem;
var tooltipVisible = false;
var hoveredHTMLEvent;
let animationTime = 5;
let withoutAnimationTime = 0;

//Flys the observer to a given location
function fly_to_location(planet, location, time) {
    window.call_native("fly_to", planet, location.longitude, location.latitude, location.height, time);
    window.call_native("print_notification", "Travelling", "to " + location.name, "send");
}

function parse_height(heightStr, unit) {
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

function parse_latitude(lat, half) {
    lat = lat.substr(0, lat.length - 1);
    if (half == 'S')
        return parseFloat(-lat);
    else
        return parseFloat(lat);
}

function parse_longitude(long, half) {
    long = long.substr(0, long.length - 1);
    if (half == 'W')
        return parseFloat(-long);
    else
        return parseFloat(long);
}

// Extracts the needed information out of the human readable place string
// and calls fly_to_location for the given location.
function travel_to(direct, planet, place, name) {
    var placeArr = place.split(" ");
    var location = {
        "longitude": parse_longitude(placeArr[0], placeArr[1]),
        "latitude": parse_latitude(placeArr[2], placeArr[3]),
        "height": parse_height(placeArr[4], placeArr[5]),
        "name": name
    };

    if (direct) {
        fly_to_location(planet, location, withoutAnimationTime);
    } else {
        fly_to_location(planet, location, animationTime);
    }
}

// Shows a tooltip if an item is hovered
function item_over_callback(properties, overview) {
    document.getElementById("event-tooltip-container").style.display = "block";
    tooltipVisible = true;
    for (var item in items._data) {
        if (items._data[item].id == properties.item) {
            document.getElementById("event-tooltip-content").innerHTML = items._data[item].content;
            document.getElementById("event-tooltip-description").innerHTML = items._data[item].description;
            document.getElementById("event-tooltip-location").innerHTML = "<i class='material-icons'>send</i> " + items._data[item].planet + " " + items._data[item].place;
            hoveredItem = items._data[item];
        }
    }
    var events = document.getElementsByClassName(properties.item);
    var event;
    for (var i = 0; i < events.length; i++) {
        if (!overview && $(events[i]).hasClass("event")) {
            event = events[i];
        } else if (overview && $(events[i]).hasClass("overviewEvent")) {
            event = events[i];
        }
    }
    hoveredHTMLEvent = event;
    hoveredHTMLEvent.classList.add('mouseOver');
    var eventRect = event.getBoundingClientRect();
    var left = eventRect.left - 150 < 0 ? 0 : eventRect.left - 150;
    document.getElementById("event-tooltip-container").style.top = eventRect.bottom + 'px';
    document.getElementById("event-tooltip-container").style.left = left + 'px';
    if (currentSpeed != paus) {
        start_redraw_tooltip(event);
    }
}

// Shows a tooltip if an item on the overview timeline is hovered
function item_over_overview_callback(properties) {
    item_over_callback(properties, true);
}

// Closes the tooltip if the mouse leaves the item and tooltip
function item_out_callback(properties) {
    if (properties.event.toElement.className != "event-tooltip") {
        document.getElementById("event-tooltip-container").style.display = "none";
        tooltipVisible = false;
        hoveredHTMLEvent.classList.remove('mouseOver');
    }
}

// Flies the observer to the location of the hovered item
function travel_to_item_location() {
    travel_to(false, hoveredItem.planet, hoveredItem.place, hoveredItem.content);
}

// Hide the tooltip if the mouse leaves the tooltip
function leave_custom_tooltip(event) {
    document.getElementById("event-tooltip-container").style.display = "none";
    tooltipVisible = false;
    hoveredHTMLEvent.classList.remove('mouseOver');
}

// Snap back items iv they were dragged with the mouse
function on_item_move_callback(item, callback) {
    callback(null);
}

function on_item_move_overview_callback(item, callback) {
    callback(null);
}

// Close the event form
function close_form() {
    parHolder.callback(null); // cancel item creation
    document.getElementById("add-event-dialog").style.display = "none";
    timeline.setOptions(editingDoneOpt);
    overviewTimeLine.setOptions(editingDoneOpt);
}

// Creates/Updates a event with the user inputs
var wrongInputStyle = "2px solid red";
function apply_event() {
    if (document.getElementById("event-dialog-name").value != ""
        && document.getElementById("event-dialog-start-date").value != ""
        && document.getElementById("event-dialog-description").value != "") {
        document.getElementById("event-dialog-name").style.border = "";
        document.getElementById("event-dialog-start-date").style.border = "";
        document.getElementById("event-dialog-description").style.border = "";
        parHolder.item.style = "border-color: " + document.getElementById("event-dialog-color").value;
        parHolder.item.content = document.getElementById("event-dialog-name").value;
        parHolder.item.start = new Date(document.getElementById("event-dialog-start-date").value);
        parHolder.item.description = document.getElementById("event-dialog-description").value;
        if (document.getElementById("event-dialog-end-date").value != "") {
            parHolder.item.end = new Date(document.getElementById("event-dialog-end-date").value);
            var diff = parHolder.item.start - parHolder.item.end;
            if (diff >= 0) {
                parHolder.item.end = null;
                document.getElementById("event-dialog-end-date").style.border = wrongInputStyle;
                return;
            } else {
                document.getElementById("event-dialog-end-date").style.border = "";
            }
        }
        parHolder.item.planet = document.getElementById("event-dialog-planet").value;
        parHolder.item.place = document.getElementById("event-dialog-location").value;
        if (parHolder.item.id == null) {
            parHolder.item.id = parHolder.item.content + parHolder.item.start + parHolder.item.end;
            parHolder.item.id = parHolder.item.id.replace(/\s/g, '');
        }
        if (parHolder.overview) {
            parHolder.item.className = 'overviewEvent ' + parHolder.item.id;
        } else {
            parHolder.item.className = 'event ' + parHolder.item.id;
        }
        parHolder.callback(parHolder.item); // send back adjusted new item
        document.getElementById("add-event-dialog").style.display = "none";
        timeline.setOptions(editingDoneOpt);
        overviewTimeLine.setOptions(editingDoneOpt);
        if (parHolder.overview) {
            parHolder.item.className = 'event ' + parHolder.item.id;
            items.update(parHolder.item);
        } else {
            parHolder.item.className = 'overviewEvent ' + parHolder.item.id;
            itemsOverview.update(parHolder.item);
        }
    } else {
        if (document.getElementById("event-dialog-name").value == "") {
            document.getElementById("event-dialog-name").style.border = wrongInputStyle;
        } else {
            document.getElementById("event-dialog-name").style.border = "";
        }
        if (document.getElementById("event-dialog-start-date").value == "") {
            document.getElementById("event-dialog-start-date").style.border = wrongInputStyle;
        } else {
            document.getElementById("event-dialog-start-date").style.border = "";
        }
        if (document.getElementById("event-dialog-description").value == "") {
            document.getElementById("event-dialog-description").style.border = wrongInputStyle;
        } else {
            document.getElementById("event-dialog-description").style.border = "";
        }
    }
}

// Called when an item is about to be updated
function on_update_callback(item, callback, overview) {
    document.getElementById("event-dialog-name").style.border = "";
    document.getElementById("event-dialog-start-date").style.border = "";
    document.getElementById("event-dialog-description").style.border = "";
    timeline.setOptions(whileEditingOpt);
    overviewTimeLine.setOptions(whileEditingOpt);
    document.getElementById("headlineForm").innerText = "Update";
    document.getElementById("add-event-dialog").style.display = "block";
    document.getElementById("event-dialog-name").value = item.content;
    document.getElementById("event-dialog-start-date").value = get_formatted_dateWithTime(item.start);
    document.getElementById("event-dialog-description").value = item.description;
    document.getElementById("event-dialog-planet").value = item.planet;
    document.getElementById("event-dialog-location").value = item.place;
    if (item.end) {
        document.getElementById("event-dialog-end-date").value = get_formatted_dateWithTime(item.end);
    } else {
        document.getElementById("event-dialog-end-date").value = "";
    }
    parHolder.item = item;
    parHolder.callback = callback;
    parHolder.overview = overview;
    set_pause();
}

// Called when an item is about to be added
function on_add_callback(item, callback, overview) {
    document.getElementById("event-dialog-name").style.border = "";
    document.getElementById("event-dialog-start-date").style.border = "";
    document.getElementById("event-dialog-description").style.border = "";
    timeline.setOptions(whileEditingOpt);
    overviewTimeLine.setOptions(whileEditingOpt);
    document.getElementById("headlineForm").innerText = "Add";
    document.getElementById("event-dialog-name").value = "";
    document.getElementById("add-event-dialog").style.display = "block";
    document.getElementById("event-dialog-start-date").value = get_formatted_dateWithTime(item.start);
    document.getElementById("event-dialog-end-date").value = "";
    document.getElementById("event-dialog-description").value = "";
    document.getElementById("event-dialog-planet").value = activePlanetCenter;
    document.getElementById("event-dialog-location").value = format_longitude(userPosition.long) + format_latitude(userPosition.lat) + format_height(userPosition.height);
    parHolder.item = item;
    parHolder.callback = callback;
    parHolder.overview = overview;
    set_pause();
}

function overview_on_update_callback(item, callback) {
    on_update_callback(item, callback, true);
}

function overview_on_add_callback(item, callback, overview) {
    on_add_callback(item, callback, true);
}

// Sets the min and max date for the timeline
function set_timeline_range(min, max) {
    var rangeOpt = {
        min: min,
        max: max
    }
    minDate = min;
    maxDate = max;
    timeline.setOptions(rangeOpt);
    overviewTimeLine.setOptions(rangeOpt);
    initial_overview_window(new Date(min), new Date(max));
}

// Sets variable values when a mouseDown event is triggered over the timeline
function mouse_down_callback() {
    timeline.setOptions(pausOpt);
    mouseOnTimelineDown = true;
    lastPlayValue = currentSpeed;
    click = true;
    mouseDownLeftTime = timeline.getWindow().start;
}

// Sets variable values when a mouseUp event is triggered over the timeline
function mouse_up_callback() {
    if (mouseOnTimelineDown && lastPlayValue != paus) {
        range.noUiSlider.set(parseInt(lastPlayValue));
    }
    mouseOnTimelineDown = false;
}

// Callbacks to differ between a Click on the overview timeline and the user dragging the overview timeline
function overviewMouse_down_callback() {
    click = true;
}

function overviewRange_change_callback() {
    click = false;
}

// Redraws the timerange indicator on the overview timeline in case the displayed time on the timeline changed
function timeline_change_callback() {
    set_overview_times();
    if (drawFocusLensCallback != null) {
        drawFocusLensCallback();
    }
}

// Redraws the timerange indicator on the overview timeline in case the displayed time on the overview timeline changed
function overview_change_callback() {
    if (drawFocusLensCallback != null) {
        drawFocusLensCallback();
    }
}

// Called when the user moves the timeline. It changes time so that the current time is alway in the middle
function range_change_callback(properties) {
    if (properties.byUser && String(properties.event) != "[object WheelEvent]") {
        if (currentSpeed != paus) {
            set_pause();
        }
        click = false;
        var dif = properties.start.getTime() - mouseDownLeftTime.getTime();
        var secondsDif = dif / 1000;
        var hoursDif = secondsDif / 60 / 60;
        var step = convert_seconds(secondsDif);
        var date = new Date(centerTime.getTime());
        date = increase_date(date, step.days, step.hours, step.minutes, step.seconds, step.milliSec);
        set_date_local(date);
        mouseDownLeftTime = new Date(properties.start.getTime());
        window.call_native("add_hours_without_animation", hoursDif);
    }
}

async function initial_overview_window(start, end) {
    overviewTimeLine.setWindow(start, end, animationFalse);
}

// Sets the custom times on the overview that represent the left and right time on the timeline
function set_overview_times() {
    overviewTimeLine.setCustomTime(timeline.getWindow().end, rightTimeId);
    overviewTimeLine.setCustomTime(timeline.getWindow().start, leftTimeId);
    overview_change_callback();
}

// Change time to the start date of the selected item
function on_select(properties) {
    mouseOverDisabled = true;
    for (var item in items._data) {
        if (items._data[item].id == properties.items) {
            var dif = items._data[item].start.getTime() - centerTime.getTime();
            var hoursDif = dif / 1000 / 60 / 60;
            if (items._data[item].start.getTimezoneOffset() > centerTime.getTimezoneOffset()) {
                hoursDif -= 1;
            } else if (items._data[item].start.getTimezoneOffset() < centerTime.getTimezoneOffset()) {
                hoursDif += 1;
            }
            window.call_native("add_hours", hoursDif);
            travel_to(true, items._data[item].planet, items._data[item].place, items._data[item].content);
        }
    }
}

// Actively redraw the snipped so if the time is paused the range indicator fades in/out together with the timeline
function redraw_focus_lens() {
    return new Promise(resolve => {
        switch (parseInt(currentSpeed)) {
            case paus:
                timeline.setOptions(pausOpt);
        }

        setTimeout(function () {
            resolve(10);
            redraw_focus_lens();
        }, redrawRate);
    });
}

async function startRedraw_focus_lens() {
    await redraw_focus_lens();
}

function add_item(start, end, id, content, style, description, planet, place) {
    var data = new Object();
    data.start = new Date(start);
    data.id = id;
    if (end != "") {
        data.end = new Date(end);
    }
    if (style != "") {
        data.style = style;
    }
    data.planet = planet;
    data.description = description;
    data.place = place;
    data.content = content;
    data.className = 'event ' + id;
    items.update(data);
    data.className = 'overviewEvent ' + id;
    itemsOverview.update(data);
}

// Change the time to the clicked value
function general_on_click(properties) {
    if (properties.what != "item" && properties.time != null) {
        var dif = properties.time.getTime() - centerTime.getTime();
        var hoursDif = dif / 1000 / 60 / 60;
        if (properties.time.getTimezoneOffset() > centerTime.getTimezoneOffset()) {
            hoursDif -= 1;
        } else if (properties.time.getTimezoneOffset() < centerTime.getTimezoneOffset()) {
            hoursDif += 1;
        }
        window.call_native("add_hours", hoursDif);
    }
}

// Called if the timeline is clicked
function on_click_callback(properties) {
    if (click) {
        general_on_click(properties);
    }
}

// Called if the overview is clicked
function on_overview_click(properties) {
    if (click) {
        general_on_click(properties);
    }
}

// Sets the time to a specific date
function set_time_to_date(date) {
    date.setHours(middleOfDay);
    window.call_native("set_date", format_date_cosmo(new Date(date.getTime())));
    var startDate = new Date(date.getTime());
    var endDate = new Date(date.getTime());
    startDate.setHours(startOfDay);
    endDate.setHours(endOfDay);
    set_pause();
    timeline.setWindow(startDate, endDate, animationFalse);
    set_overview_times();
}

function plus_one_second() {
    window.call_native("add_hours_without_animation", secondInHours);
}

function minus_one_second() {
    window.call_native("add_hours_without_animation", -secondInHours);
}

function plus_one_minute() {
    window.call_native("add_hours_without_animation", minuteInHours);
}

function minus_one_minute() {
    window.call_native("add_hours_without_animation", -minuteInHours);
}

function plus_one_hour() {
    window.call_native("add_hours_without_animation", 1);
}

function minus_one_hour() {
    window.call_native("add_hours_without_animation", -1);
}

function plus_one_day() {
    window.call_native("add_hours_without_animation", dayInHours);
}

function minus_one_day() {
    window.call_native("add_hours_without_animation", -dayInHours);
}

function plus_one_month() {
    var date = new Date(centerTime.getTime());
    centerTime.setMonth(centerTime.getMonth() + 1);
    var dif = centerTime.getTime() - date.getTime();
    var hoursDif = dif / 1000 / 60 / 60;
    window.call_native("add_hours_without_animation", hoursDif);
}

function minus_one_month() {
    var date = new Date(centerTime.getTime());
    centerTime.setMonth(centerTime.getMonth() - 1);
    var dif = centerTime.getTime() - date.getTime();
    var hoursDif = dif / 1000 / 60 / 60;
    window.call_native("add_hours_without_animation", hoursDif);
}

function plus_one_year() {
    var date = new Date(centerTime.getTime());
    centerTime.setFullYear(centerTime.getFullYear() + 1);
    var dif = centerTime.getTime() - date.getTime();
    var hoursDif = dif / 1000 / 60 / 60;
    window.call_native("add_hours_without_animation", hoursDif);
}

function minus_one_year() {
    var date = new Date(centerTime.getTime());
    centerTime.setFullYear(centerTime.getFullYear() - 1);
    var dif = centerTime.getTime() - date.getTime();
    var hoursDif = dif / 1000 / 60 / 60;
    window.call_native("add_hours_without_animation", hoursDif);
}

// Moves the displayed time window and sizes the time range according to the zoom factor
function move_window() {
    var step;
    step = convert_seconds(timelineRangeFactor);
    var startDate = new Date(centerTime.getTime());
    var endDate = new Date(centerTime.getTime());
    startDate = decrease_date(startDate, step.days, step.hours, step.minutes, step.seconds, step.milliSec);
    endDate = increase_date(endDate, step.days, step.hours, step.minutes, step.seconds, step.milliSec);
    timeline.setWindow(startDate, endDate, animationFalse);
}

function set_date(date) {
    centerTime = new Date(date);
    timeline.moveTo(centerTime, animationFalse);
    timeline.setCustomTime(centerTime, timeId);
    set_overview_times();
    document.getElementById("dateLabel").innerText = format_date_readable(centerTime);
}

// Changes the shown date to a given date without synchronizing with CosmoScout VR
function set_date_local(date) {
    centerTime = new Date(date);
    timeline.moveTo(centerTime, animationFalse);
    timeline.setCustomTime(centerTime, timeId);
    set_overview_times();
    document.getElementById("dateLabel").innerText = format_date_readable(centerTime);
}

function set_time_speed(speed) {
    $("#play-pause-icon").text("pause");
    if (speed == 0.0) {
        $("#play-pause-icon").text("play_arrow");
        set_pause();
        window.call_native("print_notification", "Pause", "Time is paused.", "pause");
    } else if (speed == 1.0) {
        window.call_native("print_notification", "Speed: Realtime", "Time runs in realtime.", "play_arrow");
    } else if (speed == 3600) {
        window.call_native("print_notification", "Speed: Hour/s", "Time runs at one hour per second.", "fast_forward");
    } else if (speed == 86400) {
        window.call_native("print_notification", "Speed: Day/s", "Time runs at one day per second.", "fast_forward");
    } else if (speed == 2628000) {
        window.call_native("print_notification", "Speed: Month/s", "Time runs at one month per second.", "fast_forward");
    } else if (speed == -1.0) {
        window.call_native("print_notification", "Speed: -Realtime", "Time runs backwards in realtime.", "fast_rewind");
    } else if (speed == -3600) {
        window.call_native("print_notification", "Speed: -Hour/s", "Time runs backwards at one hour per second.", "fast_rewind");
    } else if (speed == -86400) {
        window.call_native("print_notification", "Speed: -Day/s", "Time runs backwards at one day per second.", "fast_rewind");
    } else if (speed == -2628000) {
        window.call_native("print_notification", "Speed: -Month/s", "Time runs backwards at one month per second.", "fast_rewind");
    }

    timeSpeed = speed;
}

// Pauses the simulation
function set_pause() {
    currentSpeed = paus;
    window.call_native("set_time_speed", 0);
    document.getElementById("pause-button").innerHTML = '<i class="material-icons">play_arrow</i>';
    document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">pause</i>';
    timeline.setOptions(pausOpt);
    timelineZoomBlocked = false;
    startRedraw_focus_lens();
}

function toggle_pause() {
    if (currentSpeed != paus) {
        set_pause();
    } else {
        if (lastPlayValue == paus) {
            lastPlayValue = secForw;
        }
        range_update_callback();
    }
}

// Rewinds the simulation and increases the spedd if the simulation is already 
// running backward
function decrease_speed() {
    if (range.noUiSlider.get() > paus) {
        range.noUiSlider.set(secBack);
    } else if (currentSpeed == paus) {
        toggle_pause();
    } else {
        range.noUiSlider.set(currentSpeed - 1);
    }
}

// Increases the speed of the simulation
function increase_speed() {
    if (range.noUiSlider.get() < paus) {
        range.noUiSlider.set(secForw);
    } else if (currentSpeed == paus) {
        toggle_pause();
    } else {
        if (currentSpeed == secBack) {
            range.noUiSlider.set(secForw);
        } else {
            range.noUiSlider.set(currentSpeed - (-1));
        }
    }
}

// Called at an interaction with the slider
function range_update_callback() {
    currentSpeed = range.noUiSlider.get();
    if (firstSliderValue) {
        document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_right</i>';
        firstSliderValue = false;
        return;
    }

    document.getElementById("pause-button").innerHTML = '<i class="material-icons">pause</i>';
    timeline.setOptions(playingOpt);
    timelineZoomBlocked = true;
    if (parseInt(currentSpeed) < paus) {
        document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_left</i>';
    } else {
        document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_right</i>';
    }

    move_window(monthSpeed);

    switch (parseInt(currentSpeed)) {
        case monthBack:
            window.call_native("set_time_speed", -monthInSec);
            break;
        case dayBack:
            window.call_native("set_time_speed", -dayInSec);
            break;
        case hourBack:
            window.call_native("set_time_speed", -hourInSec);
            break;
        case secBack:
            window.call_native("set_time_speed", secBack);
            break;
        case secForw:
            window.call_native("set_time_speed", secForw);
            break;
        case hourForw:
            window.call_native("set_time_speed", hourInSec);
            break;
        case dayForw:
            window.call_native("set_time_speed", dayInSec);
            break;
        case monthForw:
            window.call_native("set_time_speed", monthInSec);
            break;
        default:
        // code block
    }
}


// Changes the size of the displayed timerange while thesimulation is still playing
function manuel_zoom_timeline(event) {
    if (timelineZoomBlocked) {
        if (event.deltaY < 0) {
            timelineRangeFactor -= timelineRangeFactor * zoomPercentage;
            if (timelineRangeFactor < minRangeFactor) {
                timelineRangeFactor = minRangeFactor;
            }
        } else {
            timelineRangeFactor += timelineRangeFactor * zoomPercentage;
            if (timelineRangeFactor > maxRangeFactor) {
                timelineRangeFactor = maxRangeFactor;
            }
        }
        range_update_callback();
    }
}

//Methods if the mouse wheel is scrolled over a time control button

function scroll_on_year(event) {
    if (event.deltaY < 0) {
        plus_one_year();
    } else {
        minus_one_year();
    }
}

function scroll_on_month(event) {
    if (event.deltaY < 0) {
        plus_one_month();
    } else {
        minus_one_month();
    }
}

function scroll_on_day(event) {
    if (event.deltaY < 0) {
        plus_one_day()
    } else {
        minus_one_day();
    }
}

function scroll_on_hour(event) {
    if (event.deltaY < 0) {
        plus_one_hour();
    } else {
        minus_one_hour()
    }
}

function scroll_on_minute(event) {
    if (event.deltaY < 0) {
        plus_one_minute();
    } else {
        minus_one_minute();
    }
}

function scroll_on_second(event) {
    if (event.deltaY < 0) {
        plus_one_second();
    } else {
        minus_one_second();
    }
}

// Resets the simulation
function reset_time() {
    overviewTimeLine.setWindow(minDate, maxDate);
    range.noUiSlider.set(secForw);
    window.call_native('reset_time')
}

timelineContainer.addEventListener("wheel", manuel_zoom_timeline, true);

document.getElementById("increase-second-button").onclick = plus_one_second;
document.getElementById("decrease-second-button").onclick = minus_one_second;

document.getElementById("increase-minute-button").onclick = plus_one_minute;
document.getElementById("decrease-minute-button").onclick = minus_one_minute;

document.getElementById("increase-hour-button").onclick = plus_one_hour;
document.getElementById("decrease-hour-button").onclick = minus_one_hour;

document.getElementById("increase-day-button").onclick = plus_one_day;
document.getElementById("decrease-day-button").onclick = minus_one_day;

document.getElementById("increase-month-button").onclick = plus_one_month;
document.getElementById("decrease-month-button").onclick = minus_one_month;

document.getElementById("increase-year-button").onclick = plus_one_year;
document.getElementById("decrease-year-button").onclick = minus_one_year;

document.getElementById("pause-button").onclick = toggle_pause;
document.getElementById("speed-decrease-button").onclick = decrease_speed;
document.getElementById("speed-increase-button").onclick = increase_speed;

document.getElementById("event-tooltip-location").onclick = travel_to_item_location;

document.getElementById("time-reset-button").onclick = reset_time;

document.getElementsByClassName('range-label')[0].addEventListener('mousedown', range_update_callback);

document.getElementById("decrease-year-button").addEventListener("wheel", scroll_on_year);
document.getElementById("decrease-month-button").addEventListener("wheel", scroll_on_month);
document.getElementById("decrease-day-button").addEventListener("wheel", scroll_on_day);
document.getElementById("decrease-hour-button").addEventListener("wheel", scroll_on_hour);
document.getElementById("decrease-minute-button").addEventListener("wheel", scroll_on_minute);
document.getElementById("decrease-second-button").addEventListener("wheel", scroll_on_second);

document.getElementById("increase-year-button").addEventListener("wheel", scroll_on_year);
document.getElementById("increase-month-button").addEventListener("wheel", scroll_on_month);
document.getElementById("increase-day-button").addEventListener("wheel", scroll_on_day);
document.getElementById("increase-hour-button").addEventListener("wheel", scroll_on_hour);
document.getElementById("increase-minute-button").addEventListener("wheel", scroll_on_minute);
document.getElementById("increase-second-button").addEventListener("wheel", scroll_on_second);

document.getElementById("event-dialog-cancel-button").onclick = close_form;
document.getElementById("event-dialog-apply-button").onclick = apply_event;

document.getElementById("event-tooltip-container").onmouseleave = leave_custom_tooltip;

// toggle if the overview by pressing the button on the right --------------------------------------

var overviewVisible = false;

function toggle_overview() {
    overviewVisible = !overviewVisible;
    document.getElementById('timeline-container').classList.toggle('overview-visible');
    if (overviewVisible) {
        document.getElementById("expand-button").innerHTML = '<i class="material-icons">expand_less</i>';
    }
    else {
        document.getElementById("expand-button").innerHTML = '<i class="material-icons">expand_more</i>';
    }
}

document.getElementById("expand-button").onclick = toggle_overview;

// toggle visibility of the increase / decrease time buttons ---------------------------------------

function mouse_enter_time_control() {
    document.getElementById("increaseControl").classList.add('mouseNear');
    document.getElementById("decreaseControl").classList.add('mouseNear');
}

function mouse_leave_time_control() {
    document.getElementById("increaseControl").classList.remove('mouseNear');
    document.getElementById("decreaseControl").classList.remove('mouseNear');
}

function enter_time_buttons() {
    document.getElementById("increaseControl").classList.add('mouseNear');
    document.getElementById("decreaseControl").classList.add('mouseNear');
}

function leave_time_buttons() {
    document.getElementById("increaseControl").classList.remove('mouseNear');
    document.getElementById("decreaseControl").classList.remove('mouseNear');
}

document.getElementById("time-control").onmouseenter = mouse_enter_time_control;
document.getElementById("time-control").onmouseleave = mouse_leave_time_control;

document.getElementById("increaseControl").onmouseenter = enter_time_buttons;
document.getElementById("increaseControl").onmouseleave = leave_time_buttons;

document.getElementById("decreaseControl").onmouseenter = enter_time_buttons;
document.getElementById("decreaseControl").onmouseleave = leave_time_buttons;

// draw the indicator which part of the overview is seen on the timeline ---------------------------

let minWidth = 30;
let offset = 2;
let shorten = 2;
let borderWidth = 3;

function drawFocusLens() {
    var leftCustomTime = document.getElementsByClassName("leftTime")[0];
    var leftRect = leftCustomTime.getBoundingClientRect();
    var rightCustomTime = document.getElementsByClassName("rightTime")[0];
    var rightRect = rightCustomTime.getBoundingClientRect();

    var divElement = document.getElementById("focus-lens");
    divElement.style.position = "absolute";
    divElement.style.left = leftRect.right + 'px';
    divElement.style.top = (leftRect.top + offset) + 'px';

    let height = leftRect.bottom - leftRect.top - shorten;
    var width = rightRect.right - leftRect.left;

    var xValue = 0;
    if (width < minWidth) {
        width = minWidth + 2 * borderWidth;
        xValue = -(leftRect.left + minWidth - rightRect.right) / 2 - borderWidth;
        divElement.style.transform = " translate(" + xValue + "px, 0px)";
    } else {
        divElement.style.transform = " translate(0px, 0px)";
    }

    divElement.style.height = height + 'px';
    divElement.style.width = width + 'px';

    divElement = document.getElementById("focus-lens-left");
    divElement.style.top = (leftRect.top + offset + height) + 'px';
    width = leftRect.right + xValue + borderWidth;
    width = width < 0 ? 0 : width;
    divElement.style.width = width + 'px';
    var body = document.getElementsByTagName("body")[0];
    var bodyRect = body.getBoundingClientRect();

    divElement = document.getElementById("focus-lens-right");
    divElement.style.top = (leftRect.top + offset + height) + 'px';
    width = bodyRect.right - rightRect.right + xValue + 1;
    width = width < 0 ? 0 : width;
    divElement.style.width = width + 'px';
}

drawFocusLensCallback = drawFocusLens;

// color picker initialization ---------------------------------------------------------------------

var picker = new CP(document.querySelector('input[type="colorPicker"]'));
picker.on("change", function (color) {
    this.source.value = '#' + color;
});

picker.on("change", function (color) {
    var colorField = document.getElementById("event-dialog-color");
    colorField.style.background = '#' + color;
});

var calenderVisible = false;
let newCenterTimeId = 0;
let newStartDateId = 1;
let newEndDateId = 2;
var state;

// calendar initialization -------------------------------------------------------------------------

// Sets the visibility of the calendar to the given value(true or false)
function set_visible(visible) {
    if (visible) {
        $('#calendar').addClass('visible');
    }
    else {
        $('#calendar').removeClass('visible');
    }
}

// Toggles the Visibility
function toggle_visible() {
    if (calenderVisible) {
        calenderVisible = false;
        set_visible(false);
    } else {
        calenderVisible = true;
        set_visible(true);
    }
}

// Called if the Calendar is used to change the date
function enter_new_center_time() {
    $('#calendar').datepicker('update', timeline.getCustomTime(timeId));
    if (calenderVisible && state == newCenterTimeId) {
        toggle_visible();
    } else if (!calenderVisible) {
        state = newCenterTimeId;
        toggle_visible();
    }
}


// Called if the Calendar is used to enter a start date of an event
function enter_start_date() {
    if (state == newStartDateId) {
        toggle_visible();
    } else {
        state = newStartDateId;
        calenderVisible = true;
        set_visible(true);
    }
}


// Called if the Calendar is used to enter the end date of an event
function enter_end_date() {
    if (state == newEndDateId) {
        toggle_visible();
    } else {
        state = newEndDateId;
        calenderVisible = true;
        set_visible(true);
    }
}

// Called if an Date in the Calendar is picked
function change_date_callback(e) {
    toggle_visible();
    switch (state) {
        case newCenterTimeId:
            set_time_to_date(e.date);
            break;
        case newStartDateId:
            document.getElementById("event-dialog-start-date").value = e.format();
            break;
        case newEndDateId:
            document.getElementById("event-dialog-end-date").value = e.format();
            break;
        default:
        // code block
    }
}

// entry point
$(document).ready(function () {
    $('#calendar').datepicker({
        weekStart: 1,
        todayHighlight: true,
        maxViewMode: 3,
        format: "yyyy-mm-dd",
        startDate: "1950-01-02",
        endDate: "2049-12-31",
    }).on("changeDate", change_date_callback);
});

document.getElementById("calendar-button").onclick = enter_new_center_time;
document.getElementById("dateLabel").onclick = enter_new_center_time;