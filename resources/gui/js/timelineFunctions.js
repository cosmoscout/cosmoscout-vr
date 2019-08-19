
var play = true;

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
let secSpeed = 0.01666666;
let hourSpeed = 60;
let daySpeed = 1440;
let monthSpeed = 43800;

let startOfDay = 0;
let middleOfDay = 12;
let endOfDay = 24;

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



var items = new vis.DataSet([
    {id: 1, content: 'item 1', start: '2014-04-20'},
    {id: 2, content: 'item 2', start: '2014-04-14'},
    {id: 3, content: 'item 3', start: '2014-04-18'},
    {id: 4, content: 'item 4', start: '2014-04-16', end: '2014-04-19'},
    {id: 5, content: 'item 5', start: '2014-04-25'},
    {id: 6, content: 'item 6', start: '2014-04-27', type: 'point'}
  ]);
// Create a Timeline
timeline = new vis.Timeline(container, items, options);
centerTime = timeline.getCurrentTime();
timeline.on('select', onSelect);
timeline.moveTo(centerTime, animationFalse);
timeline.addCustomTime(centerTime, timeId);
timeline.on('click', onClickCallback);
timeline.on('timechanged', timeChangedCallback);
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

function timeChangedCallback(properties) {
    centerTime = new Date(properties.time.getTime());
    document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);
    timeline.moveTo(centerTime, animationFalse);
    setOverviewTimes();
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
            centerTime = new Date(items._data[item].start.getTime());
            timeline.setCustomTime(centerTime, timeId);
            document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);
            timeline.moveTo(centerTime, animationFalse);
            setOverviewTimes();
        }
    }
}

function saveItems() {
    var data = items.get({
        type: {
          start: 'ISODate',
          end: 'ISODate'
        }
    });
    console.log(JSON.stringify(data, null, 2));
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
    document.getElementById("headlineForm").innerText = "Update Event";
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
    document.getElementById("headlineForm").innerText = "Add Event";
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
    if(properties.what != "item") {
        centerTime = new Date(properties.time.getTime());
        timeline.setCustomTime(centerTime, timeId);
        document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);
        timeline.moveTo(centerTime, animationFalse);
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
    centerTime = new Date(date.getTime());
    centerTime.setHours(middleOfDay);
    timeline.setCustomTime(centerTime, timeId);
    var startDate = new Date(date.getTime());
    var endDate = new Date(date.getTime());
    startDate.setHours(startOfDay);
    endDate.setHours(endOfDay);
    play = false;
    range.noUiSlider.set(0);
    timeline.setWindow(startDate, endDate, animationFalse);
    document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);
    setOverviewTimes();
}

function plusOneHour() {
    centerTime.setHours( centerTime.getHours() + 1 );
    timeline.moveTo(centerTime, animationFalse);
    timeline.setCustomTime(centerTime, timeId);
    setOverviewTimes();
    document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);
}
function minusOneHour() {
    centerTime.setHours( centerTime.getHours() - 1 );
    timeline.moveTo(centerTime, animationFalse);
    timeline.setCustomTime(centerTime, timeId);
    setOverviewTimes();
    document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);
}

function plusOneDay() {
    centerTime.setDate( centerTime.getDate() + 1 );
    timeline.moveTo(centerTime, animationFalse);
    timeline.setCustomTime(centerTime, timeId);
    setOverviewTimes();
    document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);
}
function minusOneDay() {
    centerTime.setDate( centerTime.getDate() - 1 );
    timeline.moveTo(centerTime, animationFalse);
    timeline.setCustomTime(centerTime, timeId);
    setOverviewTimes();
    document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);
}

function plusOneMonth() {
    centerTime.setMonth( centerTime.getMonth() + 1 );
    timeline.moveTo(centerTime, animationFalse);
    timeline.setCustomTime(centerTime, timeId);
    setOverviewTimes();
    document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);
}
function minusOneMonth() {
    centerTime.setMonth( centerTime.getMonth() - 1 );
    timeline.moveTo(centerTime, animationFalse);
    timeline.setCustomTime(centerTime, timeId);
    setOverviewTimes();
    document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);
}

function plusOneYear() {
    centerTime.setFullYear( centerTime.getFullYear() + 1 );
    timeline.moveTo(centerTime, animationFalse);
    timeline.setCustomTime(centerTime, timeId);
    setOverviewTimes();
    document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);
}
function minusOneYear() {
    centerTime.setFullYear( centerTime.getFullYear() - 1 );
    timeline.moveTo(centerTime, animationFalse);
    timeline.setCustomTime(centerTime, timeId);
    setOverviewTimes();
    document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);
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
    var step = convertSeconds(seconds * timelineRangeFactor);
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
  
  function makeTimeStep() {
    return new Promise(resolve => {
    let speedOpt = range.noUiSlider.get();
    switch(parseInt(speedOpt)) {
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

function showToday() {
    date = new Date(timeline.getCurrentTime().getTime());
    centerTime = new Date(date.getTime());
    timeline.setCustomTime(centerTime, timeId);
    var startDate = new Date(date.getTime());
    var endDate = new Date(date.getTime());
    startDate.setHours(startOfDay);
    endDate.setHours(endOfDay);
    play = false;
    range.noUiSlider.set(0);
    timeline.setWindow(startDate, endDate, animationFalse);
    document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);
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
