// API calls - these should be called from C++ -----------------------------------------------------

// Sets the timeline to the given date
function setDate(date) {
}

// Prints a notifivcatio for the time-speed and changes the slider if the time is paused
function setTimeSpeed(speed) {
}

// Adds a new event to the timeline
function addItem(start, end, id, content, style, description, planet, place) {
}

// Add a Button to the button bar
// @param icon The materialize icon to use
// @param tooltip Tooltip text that gets shown if the button is hovered
// @param callback Native function that gets called if the button is clicked. The function has
//                  to be registered as callback before clicking the button.
function addButton(icon, tooltip, callback) {
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
    document.getElementById("buttonControl").appendChild(button);
    $('[data-toggle="tooltip"]').tooltip({ delay: 500, placement: "top", html: false });
}

function setNorthDirection(angle) {
    $("#compass-arrow").css("transform", "rotateZ(" + angle + "rad)");
}

// Sets the active planet
function setActivePlanet(center) {
}

// Sets the position of the user
function setUserPosition(long, lat, height) {
}

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

var drawDivCallback = null;

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
    onAdd: onAddCallback,
    onUpdate: onUpdateCallback,
    onMove: onItemMoveCallback,
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
        updateTime: true,  // drag items horizontally
        updateGroup: false, // drag items from one group to another
        remove: false,       // delete an item by tapping the delete button top right
        overrideItems: false  // allow these options to override item.editable
    },
    onAdd: overviewOnAddCallback,
    onUpdate: overviewOnUpdateCallback,
    onMove: onItemMoveOverviewCallback
}

var whileEditingOpt = {
    editable: false
}

var editingDoneOpt = {
    editable: {
        add: true,         // add new items by double tapping
        updateTime: true,  // drag items horizontally
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

range.noUiSlider.on('update', rangeUpdateCallback);

var items = new vis.DataSet();
var itemsOverview = new vis.DataSet();
// Create a Timeline
timeline = new vis.Timeline(timelineContainer, items, options);
centerTime = timeline.getCurrentTime();
timeline.on('select', onSelect);
timeline.moveTo(centerTime, animationFalse);
timeline.addCustomTime(centerTime, timeId);
timeline.on('click', onClickCallback);
timeline.on('changed', timelineChangeCallback);
timeline.on('mouseDown', mouseDownCallback);
timeline.on('mouseUp', mouseUpCallback);
timeline.on('rangechange', rangechangeCallback);
timeline.on('itemover', itemoverCallback);
timeline.on('itemout', itemoutCallback);

//create overview timeline
overviewTimeLine = new vis.Timeline(overviewContainer, itemsOverview, overviewOptions);
overviewTimeLine.addCustomTime(timeline.getWindow().end, rightTimeId);
overviewTimeLine.addCustomTime(timeline.getWindow().start, leftTimeId);
overviewTimeLine.on('select', onSelect);
overviewTimeLine.on('click', onOverviewClick);
overviewTimeLine.on('changed', overviewChangeCallback);
overviewTimeLine.on('mouseDown', overviewMouseDownCallback);
overviewTimeLine.on('rangechange', overviewRangechangeCallback);
overviewTimeLine.on('itemover', itemoverOverviewCallback);
overviewTimeLine.on('itemout', itemoutCallback);
initialOverviewWindow(new Date(1950, 1), new Date(2030, 12));

document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);

moveWindow(secSpeed);

// Sets the active planet
function setActivePlanet(center) {
    activePlanetCenter = center;
}


function formatNumber(number) {
    if (Math.abs(number) < 10) return number.toFixed(2);
    else if (Math.abs(number) < 100) return number.toFixed(1);
    else return number.toFixed(0)
}

function formatHeight(height) {
    if (Math.abs(height) < 0.1) return formatNumber(height * 1000) + ' mm';
    else if (Math.abs(height) < 1) return formatNumber(height * 100) + ' cm';
    else if (Math.abs(height) < 1e4) return formatNumber(height) + ' m';
    else if (Math.abs(height) < 1e7) return formatNumber(height / 1e3) + ' km';
    else if (Math.abs(height) < 1e10) return formatNumber(height / 1e6) + ' Tsd km';
    else if (Math.abs(height / 1.496e11) < 1e4) return formatNumber(height / 1.496e11) + ' AU';
    else if (Math.abs(height / 9.461e15) < 1e3) return formatNumber(height / 9.461e15) + ' ly';
    else if (Math.abs(height / 3.086e16) < 1e3) return formatNumber(height / 3.086e16) + ' pc';

    return formatNumber(height / 3.086e19) + ' kpc';
}

function formatLatitude(lat) {
    if (lat < 0)
        return (-lat).toFixed(2) + "° S ";
    else
        return (lat).toFixed(2) + "° N ";
}

function formatLongitude(long) {
    if (long < 0)
        return (-long).toFixed(2) + "° W ";
    else
        return (long).toFixed(2) + "° E ";
}

function setUserPosition(long, lat, height) {
    userPosition.long = long;
    userPosition.lat = lat;
    userPosition.height = height;
}

// Redraws the tooltip of an event while the event is visible
function redrawTooltip(event) {
    return new Promise(resolve => {
        var eventRect = event.getBoundingClientRect();
        var left = eventRect.left - 150 < 0 ? 0 : eventRect.left - 150;
        document.getElementById("customTooltip").style.top = eventRect.bottom + 'px';
        document.getElementById("customTooltip").style.left = left + 'px';

        setTimeout(function () {
            resolve(10);
            if (tooltipVisible) {
                redrawTooltip(event);
            }
        }, redrawRate);
    });
}

// Starts redrawing the tooltip of an event
async function startRedrawTooltip(event) {
    await redrawTooltip(event);
}

var hoveredItem;
var tooltipVisible = false;
var hoveredHTMLEvent;

// Shows a tooltip if an item is hovered
function itemoverCallback(properties, overview) {
    document.getElementById("customTooltip").style.display = "block";
    tooltipVisible = true;
    for (var item in items._data) {
        if (items._data[item].id == properties.item) {
            document.getElementById("itemContent").innerHTML = items._data[item].content;
            document.getElementById("itemDescription").innerHTML = items._data[item].description;
            document.getElementById("itemLocation").innerHTML = "<i class='material-icons'>send</i> " + items._data[item].planet + " " + items._data[item].place;
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
    document.getElementById("customTooltip").style.top = eventRect.bottom + 'px';
    document.getElementById("customTooltip").style.left = left + 'px';
    if (currentSpeed != paus) {
        startRedrawTooltip(event);
    }
}

// Shows a tooltip if an item on the overview timeline is hovered
function itemoverOverviewCallback(properties) {
    itemoverCallback(properties, true);
}

// Closes the tooltip if the mouse leaves the item and tooltip
function itemoutCallback(properties) {
    if (properties.event.toElement.className != "custom-tooltip-container") {
        document.getElementById("customTooltip").style.display = "none";
        tooltipVisible = false;
        hoveredHTMLEvent.classList.remove('mouseOver');
    }
}

// Flies the observer to the location of the hovered item
function travelToItemLocation() {
    geoCode(false, hoveredItem.planet, hoveredItem.place, hoveredItem.content);
}

// Hide the tooltip if the mouse leaves the tooltip
function leaveCustomTooltip(event) {
    document.getElementById("customTooltip").style.display = "none";
    tooltipVisible = false;
    hoveredHTMLEvent.classList.remove('mouseOver');
}

// Snap back items iv they were dragged with the mouse
function onItemMoveCallback(item, callback) {
    callback(null);
}

function onItemMoveOverviewCallback(item, callback) {
    callback(null);
}

// Close the event form
function closeForm() {
    parHolder.callback(null); // cancel item creation
    document.getElementById("myForm").style.display = "none";
    timeline.setOptions(editingDoneOpt);
    overviewTimeLine.setOptions(editingDoneOpt);
}

// Creates/Updates a event with the user inputs
var wrongInputStyle = "2px solid red";
function applyEvent() {
    if (document.getElementById("eventName").value != ""
        && document.getElementById("eventStartDate").value != ""
        && document.getElementById("descriptionInput").value != "") {
        document.getElementById("eventName").style.border = "";
        document.getElementById("eventStartDate").style.border = "";
        document.getElementById("descriptionInput").style.border = "";
        parHolder.item.style = "border-color: " + document.getElementById("eventColor").value;
        parHolder.item.content = document.getElementById("eventName").value;
        parHolder.item.start = new Date(document.getElementById("eventStartDate").value);
        parHolder.item.description = document.getElementById("descriptionInput").value;
        if (document.getElementById("eventEndDate").value != "") {
            parHolder.item.end = new Date(document.getElementById("eventEndDate").value);
            var diff = parHolder.item.start - parHolder.item.end;
            if (diff >= 0) {
                parHolder.item.end = null;
                document.getElementById("eventEndDate").style.border = wrongInputStyle;
                return;
            } else {
                document.getElementById("eventEndDate").style.border = "";
            }
        }
        parHolder.item.planet = document.getElementById("planetInput").value;
        parHolder.item.place = document.getElementById("placeInput").value;
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
        document.getElementById("myForm").style.display = "none";
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
        if (document.getElementById("eventName").value == "") {
            document.getElementById("eventName").style.border = wrongInputStyle;
        } else {
            document.getElementById("eventName").style.border = "";
        }
        if (document.getElementById("eventStartDate").value == "") {
            document.getElementById("eventStartDate").style.border = wrongInputStyle;
        } else {
            document.getElementById("eventStartDate").style.border = "";
        }
        if (document.getElementById("descriptionInput").value == "") {
            document.getElementById("descriptionInput").style.border = wrongInputStyle;
        } else {
            document.getElementById("descriptionInput").style.border = "";
        }
    }
}

// Called when an item is about to be updated
function onUpdateCallback(item, callback, overview) {
    document.getElementById("eventName").style.border = "";
    document.getElementById("eventStartDate").style.border = "";
    document.getElementById("descriptionInput").style.border = "";
    timeline.setOptions(whileEditingOpt);
    overviewTimeLine.setOptions(whileEditingOpt);
    document.getElementById("headlineForm").innerText = "Update";
    document.getElementById("myForm").style.display = "block";
    document.getElementById("eventName").value = item.content;
    document.getElementById("eventStartDate").value = getFormattedDateWithTime(item.start);
    document.getElementById("descriptionInput").value = item.description;
    document.getElementById("planetInput").value = item.planet;
    document.getElementById("placeInput").value = item.place;
    if (item.end) {
        document.getElementById("eventEndDate").value = getFormattedDateWithTime(item.end);
    } else {
        document.getElementById("eventEndDate").value = "";
    }
    parHolder.item = item;
    parHolder.callback = callback;
    parHolder.overview = overview;
    setPause();
}

// Called when an item is about to be added
function onAddCallback(item, callback, overview) {
    document.getElementById("eventName").style.border = "";
    document.getElementById("eventStartDate").style.border = "";
    document.getElementById("descriptionInput").style.border = "";
    timeline.setOptions(whileEditingOpt);
    overviewTimeLine.setOptions(whileEditingOpt);
    document.getElementById("headlineForm").innerText = "Add";
    document.getElementById("eventName").value = "";
    document.getElementById("myForm").style.display = "block";
    document.getElementById("eventStartDate").value = getFormattedDateWithTime(item.start);
    document.getElementById("eventEndDate").value = "";
    document.getElementById("descriptionInput").value = "";
    document.getElementById("planetInput").value = activePlanetCenter;
    document.getElementById("placeInput").value = formatLongitude(userPosition.long) + formatLatitude(userPosition.lat) + formatHeight(userPosition.height);
    parHolder.item = item;
    parHolder.callback = callback;
    parHolder.overview = overview;
    setPause();
}

function overviewOnUpdateCallback(item, callback) {
    onUpdateCallback(item, callback, true);
}

function overviewOnAddCallback(item, callback, overview) {
    onAddCallback(item, callback, true);
}

// Sets the min and max date for the timeline
function setTimelineRange(min, max) {
    var rangeOpt = {
        min: min,
        max: max
    }
    minDate = min;
    maxDate = max;
    timeline.setOptions(rangeOpt);
    overviewTimeLine.setOptions(rangeOpt);
    initialOverviewWindow(new Date(min), new Date(max));
}

// Sets variable values when a mouseDown event is triggered over the timeline
function mouseDownCallback() {
    timeline.setOptions(pausOpt);
    mouseOnTimelineDown = true;
    lastPlayValue = currentSpeed;
    click = true;
    mouseDownLeftTime = timeline.getWindow().start;
}

// Sets variable values when a mouseUp event is triggered over the timeline
function mouseUpCallback() {
    if (mouseOnTimelineDown && lastPlayValue != paus) {
        range.noUiSlider.set(parseInt(lastPlayValue));
    }
    mouseOnTimelineDown = false;
}

// Callbacks to differ between a Click on the overview timeline and the user dragging the overview timeline
function overviewMouseDownCallback() {
    click = true;
}

function overviewRangechangeCallback() {
    click = false;
}

// Redraws the timerange indicator on the overview timeline in case the displayed time on the timeline changed
function timelineChangeCallback() {
    setOverviewTimes();
    if (drawDivCallback != null) {
        drawDivCallback();
    }
}

// Redraws the timerange indicator on the overview timeline in case the displayed time on the overview timeline changed
function overviewChangeCallback() {
    if (drawDivCallback != null) {
        drawDivCallback();
    }
}

// Called when the user moves the timeline. It changes time so that the current time is alway in the middle
function rangechangeCallback(properties) {
    if (properties.byUser && String(properties.event) != "[object WheelEvent]") {
        if (currentSpeed != paus) {
            setPause();
        }
        click = false;
        var dif = properties.start.getTime() - mouseDownLeftTime.getTime();
        var secondsDif = dif / 1000;
        var hoursDif = secondsDif / 60 / 60;
        var step = convertSeconds(secondsDif);
        var date = new Date(centerTime.getTime());
        date = increaseDate(date, step.days, step.hours, step.minutes, step.seconds, step.milliSec);
        setDateLocal(date);
        mouseDownLeftTime = new Date(properties.start.getTime());
        window.call_native("add_hours_without_animation", hoursDif);
    }
}

async function initialOverviewWindow(start, end) {
    overviewTimeLine.setWindow(start, end, animationFalse);
}

// Sets the custom times on the overview that represent the left and right time on the timeline
function setOverviewTimes() {
    overviewTimeLine.setCustomTime(timeline.getWindow().end, rightTimeId);
    overviewTimeLine.setCustomTime(timeline.getWindow().start, leftTimeId);
    overviewChangeCallback();
}

// Change time to the start date of the selected item
function onSelect(properties) {
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
            geoCode(true, items._data[item].planet, items._data[item].place, items._data[item].content);
        }
    }
}

// Actively redraw the snipped so if the time is paused the range indicator fades in/out together with the timeline
function redrawSnipped() {
    return new Promise(resolve => {
        switch (parseInt(currentSpeed)) {
            case paus:
                timeline.setOptions(pausOpt);
        }

        setTimeout(function () {
            resolve(10);
            redrawSnipped();
        }, redrawRate);
    });
}

async function startRedrawSnipped() {
    await redrawSnipped();
}

function addItem(start, end, id, content, style, description, planet, place) {
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
function generalOnClick(properties) {
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
function onClickCallback(properties) {
    if (click) {
        generalOnClick(properties);
    }
}

// Called if the overview is clicked
function onOverviewClick(properties) {
    if (click) {
        generalOnClick(properties);
    }
}

// Sets the time to a specific date
function setTimeToDate(date) {
    date.setHours(middleOfDay);
    window.call_native("setDate", formatDateCosmo(new Date(date.getTime())));
    var startDate = new Date(date.getTime());
    var endDate = new Date(date.getTime());
    startDate.setHours(startOfDay);
    endDate.setHours(endOfDay);
    setPause();
    timeline.setWindow(startDate, endDate, animationFalse);
    setOverviewTimes();
}

function plusOneSecond() {
    window.call_native("add_hours_without_animation", secondInHours);
}

function minusOneSecond() {
    window.call_native("add_hours_without_animation", -secondInHours);
}

function plusOneMinute() {
    window.call_native("add_hours_without_animation", minuteInHours);
}

function minusOneMinute() {
    window.call_native("add_hours_without_animation", -minuteInHours);
}

function plusOneHour() {
    window.call_native("add_hours_without_animation", 1);
}

function minusOneHour() {
    window.call_native("add_hours_without_animation", -1);
}

function plusOneDay() {
    window.call_native("add_hours_without_animation", dayInHours);
}

function minusOneDay() {
    window.call_native("add_hours_without_animation", -dayInHours);
}

function plusOneMonth() {
    var date = new Date(centerTime.getTime());
    centerTime.setMonth(centerTime.getMonth() + 1);
    var dif = centerTime.getTime() - date.getTime();
    var hoursDif = dif / 1000 / 60 / 60;
    window.call_native("add_hours_without_animation", hoursDif);
}

function minusOneMonth() {
    var date = new Date(centerTime.getTime());
    centerTime.setMonth(centerTime.getMonth() - 1);
    var dif = centerTime.getTime() - date.getTime();
    var hoursDif = dif / 1000 / 60 / 60;
    window.call_native("add_hours_without_animation", hoursDif);
}

function plusOneYear() {
    var date = new Date(centerTime.getTime());
    centerTime.setFullYear(centerTime.getFullYear() + 1);
    var dif = centerTime.getTime() - date.getTime();
    var hoursDif = dif / 1000 / 60 / 60;
    window.call_native("add_hours_without_animation", hoursDif);
}

function minusOneYear() {
    var date = new Date(centerTime.getTime());
    centerTime.setFullYear(centerTime.getFullYear() - 1);
    var dif = centerTime.getTime() - date.getTime();
    var hoursDif = dif / 1000 / 60 / 60;
    window.call_native("add_hours_without_animation", hoursDif);
}

// Moves the displayed time window and sizes the time range according to the zoom factor
function moveWindow() {
    var step;
    step = convertSeconds(timelineRangeFactor);
    var startDate = new Date(centerTime.getTime());
    var endDate = new Date(centerTime.getTime());
    startDate = decreaseDate(startDate, step.days, step.hours, step.minutes, step.seconds, step.milliSec);
    endDate = increaseDate(endDate, step.days, step.hours, step.minutes, step.seconds, step.milliSec);
    timeline.setWindow(startDate, endDate, animationFalse);
}

function setDate(date) {
    centerTime = new Date(date);
    timeline.moveTo(centerTime, animationFalse);
    timeline.setCustomTime(centerTime, timeId);
    setOverviewTimes();
    document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);
}

// Changes the shown date to a given date without synchronizing with CosmoScout VR
function setDateLocal(date) {
    centerTime = new Date(date);
    timeline.moveTo(centerTime, animationFalse);
    timeline.setCustomTime(centerTime, timeId);
    setOverviewTimes();
    document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);
}

function setTimeSpeed(speed) {
    $("#play-pause-icon").text("pause");
    if (speed == 0.0) {
        $("#play-pause-icon").text("play_arrow");
        setPause();
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
function setPause() {
    currentSpeed = paus;
    window.call_native("setTimeSpeed", 0);
    document.getElementById("btnPause").innerHTML = '<i class="material-icons">play_arrow</i>';
    document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">pause</i>';
    timeline.setOptions(pausOpt);
    timelineZoomBlocked = false;
    startRedrawSnipped();
}

function togglePause() {
    if (currentSpeed != paus) {
        setPause();
    } else {
        if (lastPlayValue == paus) {
            lastPlayValue = secForw;
        }
        rangeUpdateCallback();
    }
}

// Rewinds the simulation and increases the spedd if the simulation is already 
// running backward
function decreaseSpeed() {
    if (range.noUiSlider.get() > paus) {
        range.noUiSlider.set(secBack);
    } else if (currentSpeed == paus) {
        togglePause();
    } else {
        range.noUiSlider.set(currentSpeed - 1);
    }
}

// Increases the speed of the simulation
function increaseSpeed() {
    if (range.noUiSlider.get() < paus) {
        range.noUiSlider.set(secForw);
    } else if (currentSpeed == paus) {
        togglePause();
    } else {
        if (currentSpeed == secBack) {
            range.noUiSlider.set(secForw);
        } else {
            range.noUiSlider.set(currentSpeed - (-1));
        }
    }
}

// Called at an interaction with the slider
function rangeUpdateCallback() {
    currentSpeed = range.noUiSlider.get();
    if (firstSliderValue) {
        document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_right</i>';
        firstSliderValue = false;
        return;
    }

    document.getElementById("btnPause").innerHTML = '<i class="material-icons">pause</i>';
    timeline.setOptions(playingOpt);
    timelineZoomBlocked = true;
    if (parseInt(currentSpeed) < paus) {
        document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_left</i>';
    } else {
        document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_right</i>';
    }

    moveWindow(monthSpeed);

    switch (parseInt(currentSpeed)) {
        case monthBack:
            window.call_native("setTimeSpeed", -monthInSec);
            break;
        case dayBack:
            window.call_native("setTimeSpeed", -dayInSec);
            break;
        case hourBack:
            window.call_native("setTimeSpeed", -hourInSec);
            break;
        case secBack:
            window.call_native("setTimeSpeed", secBack);
            break;
        case secForw:
            window.call_native("setTimeSpeed", secForw);
            break;
        case hourForw:
            window.call_native("setTimeSpeed", hourInSec);
            break;
        case dayForw:
            window.call_native("setTimeSpeed", dayInSec);
            break;
        case monthForw:
            window.call_native("setTimeSpeed", monthInSec);
            break;
        default:
        // code block
    }
}


// Changes the size of the displayed timerange while thesimulation is still playing
function manuelZoomTimeline(event) {
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
        rangeUpdateCallback();
    }
}

//Methods if the mouse wheel is scrolled over a time control button

function scrollOnYear(event) {
    if (event.deltaY < 0) {
        plusOneYear();
    } else {
        minusOneYear();
    }
}

function scrollOnMonth(event) {
    if (event.deltaY < 0) {
        plusOneMonth();
    } else {
        minusOneMonth();
    }
}

function scrollOnDay(event) {
    if (event.deltaY < 0) {
        plusOneDay()
    } else {
        minusOneDay();
    }
}

function scrollOnHour(event) {
    if (event.deltaY < 0) {
        plusOneHour();
    } else {
        minusOneHour()
    }
}

function scrollOnMinute(event) {
    if (event.deltaY < 0) {
        plusOneMinute();
    } else {
        minusOneMinute();
    }
}

function scrollOnSecond(event) {
    if (event.deltaY < 0) {
        plusOneSecond();
    } else {
        minusOneSecond();
    }
}

// Resets the simulation
function resetTime() {
    overviewTimeLine.setWindow(minDate, maxDate);
    range.noUiSlider.set(secForw);
    window.call_native('reset_time')
}

timelineContainer.addEventListener("wheel", manuelZoomTimeline, true);

document.getElementById("btnIncreaseSecond").onclick = plusOneSecond;
document.getElementById("btnDecreaseSecond").onclick = minusOneSecond;

document.getElementById("btnIncreaseMinute").onclick = plusOneMinute;
document.getElementById("btnDecreaseMinute").onclick = minusOneMinute;

document.getElementById("btnIncreaseHour").onclick = plusOneHour;
document.getElementById("btnDecreaseHour").onclick = minusOneHour;

document.getElementById("btnIncreaseDay").onclick = plusOneDay;
document.getElementById("btnDecreaseDay").onclick = minusOneDay;

document.getElementById("btnIncreaseMonth").onclick = plusOneMonth;
document.getElementById("btnDecreaseMonth").onclick = minusOneMonth;

document.getElementById("btnIncreaseYear").onclick = plusOneYear;
document.getElementById("btnDecreaseYear").onclick = minusOneYear;

document.getElementById("btnPause").onclick = togglePause;
document.getElementById("btnDecreaseSpeed").onclick = decreaseSpeed;
document.getElementById("btnIncreaseSpeed").onclick = increaseSpeed;

document.getElementById("itemLocation").onclick = travelToItemLocation;

document.getElementById("btnReset").onclick = resetTime;

document.getElementsByClassName('range-label')[0].addEventListener('mousedown', rangeUpdateCallback);

document.getElementById("btnDecreaseYear").addEventListener("wheel", scrollOnYear);
document.getElementById("btnDecreaseMonth").addEventListener("wheel", scrollOnMonth);
document.getElementById("btnDecreaseDay").addEventListener("wheel", scrollOnDay);
document.getElementById("btnDecreaseHour").addEventListener("wheel", scrollOnHour);
document.getElementById("btnDecreaseMinute").addEventListener("wheel", scrollOnMinute);
document.getElementById("btnDecreaseSecond").addEventListener("wheel", scrollOnSecond);

document.getElementById("btnIncreaseYear").addEventListener("wheel", scrollOnYear);
document.getElementById("btnIncreaseMonth").addEventListener("wheel", scrollOnMonth);
document.getElementById("btnIncreaseDay").addEventListener("wheel", scrollOnDay);
document.getElementById("btnIncreaseHour").addEventListener("wheel", scrollOnHour);
document.getElementById("btnIncreaseMinute").addEventListener("wheel", scrollOnMinute);
document.getElementById("btnIncreaseSecond").addEventListener("wheel", scrollOnSecond);

document.getElementById("btnCancel").onclick = closeForm;
document.getElementById("btnApply").onclick = applyEvent;

document.getElementById("customTooltip").onmouseleave = leaveCustomTooltip;

// toggle if the overview by pressing the button on the right --------------------------------------

var overviewVisible = false;

function toggleOverview() {
    overviewVisible = !overviewVisible;
    document.getElementById('timelineContainer').classList.toggle('visible');
    if (overviewVisible) {
        document.getElementById("btnExpand").innerHTML = '<i class="material-icons">expand_less</i>';
    }
    else {
        document.getElementById("btnExpand").innerHTML = '<i class="material-icons">expand_more</i>';
    }
}

document.getElementById("btnExpand").onclick = toggleOverview;

// toggle visibility of the increase / decrease time buttons ---------------------------------------

function mouseEnterTimeControl() {
    document.getElementById("increaseControl").classList.add('mouseNear');
    document.getElementById("decreaseControl").classList.add('mouseNear');
}

function mouseLeaveTimeControl() {
    document.getElementById("increaseControl").classList.remove('mouseNear');
    document.getElementById("decreaseControl").classList.remove('mouseNear');
}

function enterTimeButtons() {
    document.getElementById("increaseControl").classList.add('mouseNear');
    document.getElementById("decreaseControl").classList.add('mouseNear');
}

function leaveTimeButtons() {
    document.getElementById("increaseControl").classList.remove('mouseNear');
    document.getElementById("decreaseControl").classList.remove('mouseNear');
}

document.getElementById("timeControl").onmouseenter = mouseEnterTimeControl;
document.getElementById("timeControl").onmouseleave = mouseLeaveTimeControl;

document.getElementById("increaseControl").onmouseenter = enterTimeButtons;
document.getElementById("increaseControl").onmouseleave = leaveTimeButtons;

document.getElementById("decreaseControl").onmouseenter = enterTimeButtons;
document.getElementById("decreaseControl").onmouseleave = leaveTimeButtons;

// draw the indicator which part of the overview is seen on the timeline ---------------------------

let minWidth = 30;
let offset = 2;
let shorten = 2;
let borderWidth = 3;

function drawDiv() {
    var leftCustomTime = document.getElementsByClassName("leftTime")[0];
    var leftRect = leftCustomTime.getBoundingClientRect();
    var rightCustomTime = document.getElementsByClassName("rightTime")[0];
    var rightRect = rightCustomTime.getBoundingClientRect();

    var divElement = document.getElementById("focusLens");
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

    divElement = document.getElementById("focusLensLeft");
    divElement.style.top = (leftRect.top + offset + height) + 'px';
    width = leftRect.right + xValue + borderWidth;
    width = width < 0 ? 0 : width;
    divElement.style.width = width + 'px';
    var body = document.getElementsByTagName("body")[0];
    var bodyRect = body.getBoundingClientRect();

    divElement = document.getElementById("focusLensRight");
    divElement.style.top = (leftRect.top + offset + height) + 'px';
    width = bodyRect.right - rightRect.right + xValue + 1;
    width = width < 0 ? 0 : width;
    divElement.style.width = width + 'px';
}

drawDivCallback = drawDiv;

// color picker initialization ---------------------------------------------------------------------

var picker = new CP(document.querySelector('input[type="colorPicker"]'));
picker.on("change", function (color) {
    this.source.value = '#' + color;
});

picker.on("change", function (color) {
    var colorField = document.getElementById("eventColor");
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
function enterNewCenterTime() {
    $('#calendar').datepicker('update', timeline.getCustomTime(timeId));
    if (calenderVisible && state == newCenterTimeId) {
        toggle_visible();
    } else if (!calenderVisible) {
        state = newCenterTimeId;
        toggle_visible();
    }
}


// Called if the Calendar is used to enter a start date of an event
function enterStartDate() {
    if (state == newStartDateId) {
        toggle_visible();
    } else {
        state = newStartDateId;
        calenderVisible = true;
        set_visible(true);
    }
}


// Called if the Calendar is used to enter the end date of an event
function enterEndDate() {
    if (state == newEndDateId) {
        toggle_visible();
    } else {
        state = newEndDateId;
        calenderVisible = true;
        set_visible(true);
    }
}

// Called if an Date in the Calendar is picked
function changeDateCallback(e) {
    toggle_visible();
    switch (state) {
        case newCenterTimeId:
            setTimeToDate(e.date);
            break;
        case newStartDateId:
            document.getElementById("eventStartDate").value = e.format();
            break;
        case newEndDateId:
            document.getElementById("eventEndDate").value = e.format();
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
    }).on("changeDate", changeDateCallback);
});

document.getElementById("btnCalendar").onclick = enterNewCenterTime;
document.getElementById("dateLabel").onclick = enterNewCenterTime;