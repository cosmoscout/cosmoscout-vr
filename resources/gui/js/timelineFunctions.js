
var play = true;

let monthInSec = 2628000;
let dayInSec = 86400;
let hourInSec =  3600;
let minuteInSec = 60;

let timeId = 0;

let leftTimeId = 'leftTime';
let rightTimeId = 'rightTime';

var parHolder = new Object();

var drawDivCallback = null;

var firstTime = true;

let zoomPercentage = 0.5;
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

var timelineRangeFactor = 6000;
let maxRangeFactor = 60000;
let minRangeFactor = 500;

let redrawRate = 16.666666;
let secSpeed = 0;
let hourSpeed = 60;
let daySpeed = 1440;
let monthSpeed = 43800;

let startOfDay = 0;
let middleOfDay = 12;
let endOfDay = 24;

var currentSpeed;

// Create a DataSet (allows two way data-binding)
var items;

var timeline;
var overviewTimeLine;
var centerTime;

// DOM element where the Timeline will be attached
var container = document.getElementById('visualization');
var overviewContainer = document.getElementById('overview');

// Configuration for the Timeline
var options = {
    minHeight: 90,
    maxHeight: 90,
    stack: false,
    max: new Date(2030, 12),
    min: new Date(1950,1),
    zoomable: false,
    moveable: false,
    showCurrentTime: false,
    editable: {
        add: true,         // add new items by double tapping
        updateTime: true,  // drag items horizontally
        updateGroup: false, // drag items from one group to another
        remove: true,       // delete an item by tapping the delete button top right
        overrideItems: false  // allow these options to override item.editable
    },
    onAdd: onAddCallback,
    onUpdate: onUpdateCallback,
    
};

var playingOpt = {
    moveable: false
}

var pausOpt = {
    moveable: true
}

var overviewOptions = {
    minHeight: 30,
    maxHeight: 30,
    stack: false,
    max: new Date(2030, 12),
    min: new Date(1950,1),
    zoomable: true,
    moveable: true,
    showCurrentTime: false,
    editable: false,
}

var animationFalse = {
    animation: false
};

var whileEditingOpt = {
    editable: false
}

var editingDoneOpt = {
    editable: {
        add: true,         // add new items by double tapping
        updateTime: true,  // drag items horizontally
        updateGroup: false, // drag items from one group to another
        remove: true,       // delete an item by tapping the delete button top right
        overrideItems: false  // allow these options to override item.editable
    }
}

var lastPlayValue = secForw;
var mouseOnTimelineDown = false;

//create range
var range = document.getElementById('range');

noUiSlider.create(range, {
    range: {
        'min': monthBack,
        '12,5%' : dayBack,
        '25%' : hourBack,
        '37,5%' : secBack,
        '50%' : paus,
        '62,5%' : secForw,
        '75%' : hourForw,
        '87,5%' : dayForw,
        'max' : monthForw,
    },
    snap: true,
    start: secForw,
    pips: {
        mode: 'values',
        values: [],
        density: 12.5
    },
    
});

range.noUiSlider.on('update', rangeUpdateCallback);

var items = new vis.DataSet();
// Create a Timeline
timeline = new vis.Timeline(container, items, options);
centerTime = timeline.getCurrentTime();
timeline.on('select', onSelect);
timeline.moveTo(centerTime, animationFalse);
timeline.addCustomTime(centerTime, timeId);
timeline.on('click', onClickCallback);
timeline.on('changed', timelineChangeCallback);
timeline.on('mouseDown', mouseDownCallback);
timeline.on('mouseUp', mouseUpCallback);

//create overview timeline
overviewTimeLine = new vis.Timeline(overviewContainer, items, overviewOptions);
overviewTimeLine.addCustomTime(timeline.getWindow().end, rightTimeId);
overviewTimeLine.addCustomTime(timeline.getWindow().start, leftTimeId);
overviewTimeLine.on('select', onSelect);
overviewTimeLine.on('click', onOverviewClick);
overviewTimeLine.on('changed', overviewChangeCallback);
initialOverviewWindow(new Date(1950,1), new Date(2030, 12));

document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);

startTimeSteps();


function setTimelineRange(min, max) {
    var rangeOpt = {
        min: min,
        max: max
    }
    timeline.setOptions(rangeOpt);
}

function mouseDownCallback(properties) {
    mouseOnTimelineDown = true;
    lastPlayValue = range.noUiSlider.get();
    range.noUiSlider.set(paus);
    timeline.setOptions(pausOpt);
}

function mouseUpCallback(properties) {
    if(mouseOnTimelineDown) {
        range.noUiSlider.set(parseInt(lastPlayValue));
    }
    mouseOnTimelineDown = false;
}

function timelineChangeCallback() {
    setOverviewTimes();
    if(drawDivCallback != null) {
        drawDivCallback();
    }
}

function overviewChangeCallback() {
    if(drawDivCallback != null) {
        drawDivCallback();
    }
}

async function initialOverviewWindow(start, end) {
    overviewTimeLine.setWindow(start, end, animationFalse);
}

function setOverviewTimes() {
    overviewTimeLine.setCustomTime(timeline.getWindow().end, rightTimeId);
    overviewTimeLine.setCustomTime(timeline.getWindow().start, leftTimeId);
    overviewChangeCallback();
}

function closeForm() {
    parHolder.callback(null); // cancel item creation
    document.getElementById("myForm").style.display = "none";
    timeline.setOptions(editingDoneOpt);
}

function onSelect (properties) {
    mouseOverDisabled = true;
    for(var item in items._data) {
        if(items._data[item].id == properties.items) {
            window.call_native("set_date", formatDateCosmo(new Date(items._data[item].start.getTime())));
            setOverviewTimes();
        }
    }
}

function set_items(events) {
    var data = JSON.parse(events);
    for(item in data) {
        data[item].start = new Date(data[item].start);
    }
    items.clear();
    items.add(data);
}

function saveItems() {
    var data = items.get({
        type: {
          start: 'ISODate',
          end: 'ISODate'
        }
    });
}

function applyEvent() {  
    if (document.getElementById("eventName").value != ""
    && document.getElementById("eventStartDate").value != "") {
        parHolder.item.style = "background-color: " + document.getElementById("eventColor").value;
        parHolder.item.content = document.getElementById("eventName").value;
        parHolder.item.start = new Date(document.getElementById("eventStartDate").value);
        if(document.getElementById("eventEndDate").value != "") {
            parHolder.item.end = new Date(document.getElementById("eventEndDate").value);
            
        }
        parHolder.callback(parHolder.item); // send back adjusted new item
        document.getElementById("myForm").style.display = "none";
        timeline.setOptions(editingDoneOpt);
        saveItems();
    }
}

function onUpdateCallback(item, callback) {
    play = false;
    timeline.setOptions(whileEditingOpt);
    document.getElementById("headlineForm").innerText = "Update";
    document.getElementById("myForm").style.display = "block";
    document.getElementById("eventName").value = item.content;
    document.getElementById("eventStartDate").value = getFormattedDate(item.start);
    if(item.end) {
        document.getElementById("eventEndDate").value = getFormattedDate(item.end);
    } else {
        document.getElementById("eventEndDate").value = "";
    }
    parHolder.item = item;
    parHolder.callback = callback;
    play = false;
    range.noUiSlider.set(paus);
}

function onAddCallback(item, callback) {
    play = false;
    timeline.setOptions(whileEditingOpt);
    document.getElementById("headlineForm").innerText = "Add";
    document.getElementById("eventName").value = "";
    document.getElementById("myForm").style.display = "block";
    document.getElementById("eventStartDate").value = getFormattedDate(item.start);
    document.getElementById("eventEndDate").value = "";
    parHolder.item = item;
    parHolder.callback = callback;
    play = false;
    range.noUiSlider.set(paus);
}


function generalOnClick(properties) {
    if(properties.what != "item" && properties.what != null) {
        window.call_native("set_date", formatDateCosmo(new Date(properties.time.getTime())));
        setOverviewTimes();
    }
}

function onClickCallback(properties) {
    generalOnClick(properties);
}

function onOverviewClick (properties){
    generalOnClick(properties);
}


function setTimeToDate(date) {
    date.setHours(middleOfDay);
    window.call_native("set_date", formatDateCosmo(new Date(date.getTime())));
    var startDate = new Date(date.getTime());
    var endDate = new Date(date.getTime());
    startDate.setHours(startOfDay);
    endDate.setHours(endOfDay);
    play = false;
    range.noUiSlider.set(0);
    timeline.setWindow(startDate, endDate, animationFalse);
    setOverviewTimes();
}

function plusOneHour() {
    window.call_native("add_hours", 1);
}
function minusOneHour() {
    window.call_native("add_hours", -1);
}

function plusOneDay() {
    window.call_native("add_hours", 24);
    
}
function minusOneDay() {
    window.call_native("add_hours", -24);
}

function plusOneMonth() {
    window.call_native("add_hours", 730);
}
function minusOneMonth() {
    window.call_native("add_hours", -730);
}

function plusOneYear() {
    window.call_native("add_hours", 8760);
}
function minusOneYear() {
    window.call_native("add_hours", -8760);
}

function increaseCenterTime(days, hours, minutes, seconds, milliSec) {
    centerTime = increaseDate(centerTime, days, hours, minutes, seconds, milliSec);
    timeline.setCustomTime(centerTime, timeId);
    setOverviewTimes();
    document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);
}

function decreaseCenterTime(days, hours, minutes, seconds, milliSec) {
    centerTime = decreaseDate(centerTime, days, hours, minutes, seconds, milliSec);
    timeline.setCustomTime(centerTime, timeId);
    setOverviewTimes();
    document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);
}

function moveCustomTime(seconds, forward) {
    play = true;
    var converted = convertSeconds(seconds);
    if(forward) {
        increaseCenterTime(converted.days, converted.hours, converted.minutes, converted.seconds, converted.milliSec);
    } else {
        decreaseCenterTime(converted.days, converted.hours, converted.minutes, converted.seconds, converted.milliSec);
    }
    var step;
    if(seconds == secSpeed)
        seconds++;
    step = convertSeconds(seconds * timelineRangeFactor);
    var startDate = new Date(centerTime.getTime());
    var endDate = new Date(centerTime.getTime());
    startDate = decreaseDate(startDate, step.days, step.hours, step.minutes, step.seconds, step.milliSec);
    endDate = increaseDate(endDate, step.days, step.hours, step.minutes, step.seconds, step.milliSec);
    if(firstTime) {
        firstTime = false;
        timeline.setWindow(startDate, endDate);
    } else {
        timeline.setWindow(startDate, endDate, animationFalse);
    }
}
  
function set_date(date) {
    centerTime = new Date(date);
    timeline.moveTo(centerTime, animationFalse);
    timeline.setCustomTime(centerTime, timeId);
    setOverviewTimes();
    document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);
}

function makeTimeStep() {
    return new Promise(resolve => {
    switch(parseInt(currentSpeed)) {
        case monthBack:
            moveCustomTime(monthSpeed, false);
            timeline.setOptions(playingOpt);
            timelineZoomBlocked = true;
          break;
        case dayBack:
            moveCustomTime(daySpeed, false);
            timeline.setOptions(playingOpt);
            timelineZoomBlocked = true;
          break;
        case hourBack:
            moveCustomTime(hourSpeed, false);
            timeline.setOptions(playingOpt);
            timelineZoomBlocked = true;
          break;
        case secBack:
            moveCustomTime(secSpeed, false);
            timeline.setOptions(playingOpt);
            timelineZoomBlocked = true;
            break;
        case paus:
            play = false;
            timeline.setOptions(pausOpt);
            timelineZoomBlocked = false;
         break; 
        case secForw:
            moveCustomTime(secSpeed, true);
            timeline.setOptions(playingOpt);
            timelineZoomBlocked = true;
            break;
        case hourForw:
            moveCustomTime(hourSpeed, true);
            timeline.setOptions(playingOpt);
            timelineZoomBlocked = true;
          break;
        case dayForw:
            moveCustomTime(daySpeed, true);
            timeline.setOptions(playingOpt);
            timelineZoomBlocked = true;
          break;
        case monthForw:
            moveCustomTime(monthSpeed, true)
            timeline.setOptions(playingOpt);
            timelineZoomBlocked = true;
            break;       
        default:
          // code block
    }

    setTimeout(function() {
        resolve(10);
        makeTimeStep();
        }, redrawRate);
    });
  }

  async function startTimeSteps() {
    await makeTimeStep();
  }


function rangeUpdateCallback(values, handle, unencoded, tap, positions) {
    currentSpeed = range.noUiSlider.get();
    if(firstSliderValue) {
        firstSliderValue = false;
        return;
    }
    switch(parseInt(currentSpeed)) {
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
        case paus:
            window.call_native("set_time_speed", 0);
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

function showToday() {
    date = new Date(timeline.getCurrentTime().getTime());
    window.call_native("set_date", formatDateCosmo(new Date(date.getTime())));
    var startDate = new Date(date.getTime());
    var endDate = new Date(date.getTime());
    startDate.setHours(startOfDay);
    endDate.setHours(endOfDay);
    play = false;
    range.noUiSlider.set(0);
    timeline.setWindow(startDate, endDate, animationFalse);
    setOverviewTimes();
}

function manuelZoomTimeline(event) {
    if(timelineZoomBlocked) {
        if(event.deltaY < 0) {
            timelineRangeFactor -= timelineRangeFactor * zoomPercentage;
            if(timelineRangeFactor < minRangeFactor) {
                timelineRangeFactor = minRangeFactor;
            }
        } else {
            timelineRangeFactor += timelineRangeFactor * zoomPercentage;
            if(timelineRangeFactor > maxRangeFactor) {
                timelineRangeFactor = maxRangeFactor;
            }
        }
    } else {
        if(event.deltaY < 0) {
            timeline.zoomIn(zoomPercentage, animationFalse);
        } else {
            timeline.zoomOut(zoomPercentage, animationFalse);
        }
    }
}

function manuelZoomOverview(event) {
    if(event.deltaY < 0) {
        overviewTimeLine.zoomIn(zoomPercentage, animationFalse);
    } else {
        overviewTimeLine.zoomOut(zoomPercentage, animationFalse);
    }
}

container.addEventListener("wheel", manuelZoomTimeline, true);
overviewContainer.addEventListener("wheel", manuelZoomOverview, true);

document.getElementById("plusOneHour").onclick = plusOneHour;
document.getElementById("minusOneHour").onclick = minusOneHour;

document.getElementById("plusOneDay").onclick = plusOneDay;
document.getElementById("minusOneDay").onclick = minusOneDay;

document.getElementById("plusOneMonth").onclick = plusOneMonth;
document.getElementById("minusOneMonth").onclick = minusOneMonth;

document.getElementById("plusOneYear").onclick = plusOneYear;
document.getElementById("minusOneYear").onclick = minusOneYear;

document.getElementById("btnCancel").onclick = closeForm;
document.getElementById("btnApply").onclick = applyEvent;

document.getElementById("btnToday").onclick = showToday;


//Prevents scrolling on page
function preventDefault(e) {
    e = e || window.event;
    if (e.preventDefault)
        e.preventDefault();
    e.returnValue = false;  
}
window.onwheel = preventDefault;
