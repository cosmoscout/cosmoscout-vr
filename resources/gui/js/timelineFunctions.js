
var play = true;

let monthInSec = 2628000;
let dayInSec = 86400;
let hourInSec =  3600;
let minuteInSec = 60;

let timeId = 0;

let leftTimeId = 'leftTime';
let rightTimeId = 'rightTime';

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
let secSpeed = 0.0166666;
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

var click;
var mouseDownLeftTime;
var inRangeChange = false;

// DOM element where the Timeline will be attached
var container = document.getElementById('visualization');
var overviewContainer = document.getElementById('overview');

// Configuration for the Timeline
var options = {
    minHeight: 25,
    maxHeight: 25,
    stack: false,
    max: new Date(2030, 12),
    min: new Date(1950,1),
    zoomable: false,
    moveable: false,
    showCurrentTime: false,
    editable: false
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
    min: new Date(1950,1),
    zoomable: true,
    moveable: true,
    showCurrentTime: false,
    editable: false,
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
        '6%' : dayBack,
        '12%' : hourBack,
        '18%' : secBack,
        '50%' : paus,
        '82%' : secForw,
        '88%' : hourForw,
        '94%' : dayForw,
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
timeline.on('rangechange', rangechangeCallback);

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
    click = true;
    mouseDownLeftTime = timeline.getWindow().start;
}

function mouseUpCallback(properties) {
    if(mouseOnTimelineDown) {
        range.noUiSlider.set(parseInt(lastPlayValue));
    }
    mouseOnTimelineDown = false;
    inRangeChange = false;
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

function rangechangeCallback(properties) {
    if(properties.byUser && String(properties.event) != "[object WheelEvent]") {
        inRangeChange = true;
        click = false;
        var dif = mouseDownLeftTime.getTime() - properties.start.getTime();
        var secondsDif = dif / 1000;
        var step = convertSeconds(secondsDif);
        var date = new Date(centerTime.getTime());
        date = decreaseDate(date, step.days, step.hours, step.minutes, step.seconds, step.milliSec);
        set_date_local(date);
        mouseDownLeftTime = new Date(properties.start.getTime());
        window.call_native("set_date_direct", formatDateCosmo(date));
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

function add_item(start, end, id, content, style, description) {
    var data = new Object();
    data.start = new Date(start);
    data.id = id;
    if(end != "") {
        data.end = end;
    }
    if(style != "") {
        data.style = style;
    }
    item = document.createElement('div');
    item.setAttribute('class', 'tooltipped');
    item.setAttribute('data-position', 'bottom');
    item.setAttribute('data-tooltip', description);
    item.appendChild(document.createTextNode(content));
    data.content = item;
    data.className = 'tooltipped overview';
    items.update(data);
    var events = document.getElementsByClassName('tooltipped')
    for(var i=0; i<events.length; i++) {
        if(events[i].textContent == content && $(events[i]).hasClass("overview")) {
            events[i].setAttribute('data-position', 'top');
            events[i].setAttribute('data-tooltip', content);
        }
    }
    $('.tooltipped').tooltip({'enterDelay':500, 'margin':-8});
}

function generalOnClick(properties) {
    if(properties.what != "item" && properties.time != null) {
        window.call_native("set_date", formatDateCosmo(new Date(properties.time.getTime())));
        setOverviewTimes();
    }
}

function onClickCallback(properties) {
    if(!click) {
        return;
    }
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

function plusOneMinute() {
    window.call_native("add_hours", 0.01666666666666);
}

function minusOneMinute() {
    window.call_native("add_hours", -0.01666666666666);
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
    centerTime.setMonth(centerTime.getMonth() + 1);
    window.call_native("set_date", formatDateCosmo(new Date(centerTime.getTime())));
}
function minusOneMonth() {
    centerTime.setMonth(centerTime.getMonth() - 1);
    window.call_native("set_date", formatDateCosmo(new Date(centerTime.getTime())));
}

function plusOneYear() {
    centerTime.setFullYear(centerTime.getFullYear() + 1);
    window.call_native("set_date", formatDateCosmo(new Date(centerTime.getTime())));
}
function minusOneYear() {
    centerTime.setFullYear(centerTime.getFullYear() - 1);
    window.call_native("set_date", formatDateCosmo(new Date(centerTime.getTime())));
}



function moveWindow(seconds) {
    play = true;
    var step;
    step = convertSeconds(seconds * timelineRangeFactor);
    var startDate = new Date(centerTime.getTime());
    var endDate = new Date(centerTime.getTime());
    startDate = decreaseDate(startDate, step.days, step.hours, step.minutes, step.seconds, step.milliSec);
    endDate = increaseDate(endDate, step.days, step.hours, step.minutes, step.seconds, step.milliSec);
    timeline.setWindow(startDate, endDate, animationFalse);
}
  
function set_date(date) {
    if(inRangeChange) {
        return;
    }
    centerTime = new Date(date);
    timeline.moveTo(centerTime, animationFalse);
    timeline.setCustomTime(centerTime, timeId);
    setOverviewTimes();
    document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);
}

function set_date_local(date) {
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
            moveWindow(monthSpeed);
            timeline.setOptions(playingOpt);
            timelineZoomBlocked = true;
          break;
        case dayBack:
            moveWindow(daySpeed);
            timeline.setOptions(playingOpt);
            timelineZoomBlocked = true;
          break;
        case hourBack:
            moveWindow(hourSpeed);
            timeline.setOptions(playingOpt);
            timelineZoomBlocked = true;
          break;
        case secBack:
            moveWindow(secSpeed);
            timeline.setOptions(playingOpt);
            timelineZoomBlocked = true;
            break;
        case paus:
            play = false;
            timeline.setOptions(pausOpt);
            timelineZoomBlocked = false;
         break; 
        case secForw:
            moveWindow(secSpeed);
            timeline.setOptions(playingOpt);
            timelineZoomBlocked = true;
            break;
        case hourForw:
            moveWindow(hourSpeed);
            timeline.setOptions(playingOpt);
            timelineZoomBlocked = true;
          break;
        case dayForw:
            moveWindow(daySpeed);
            timeline.setOptions(playingOpt);
            timelineZoomBlocked = true;
          break;
        case monthForw:
            moveWindow(monthSpeed)
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

function set_time_speed(speed) {
    if(speed == paus) {
        range.noUiSlider.set(paus);
    }
}

function togglePaus() {
    if(play) {
        lastPlayValue = range.noUiSlider.get();;
        range.noUiSlider.set(paus);
    } else {
        if(lastPlayValue == paus) {
            lastPlayValue = secForw;
        }
        range.noUiSlider.set(parseInt(lastPlayValue));
    }
}

function decreaseSpeed() {
    range.noUiSlider.set(currentSpeed-1);
}

function increaseSpeed() {
    range.noUiSlider.set(currentSpeed-(-1));
}

function rangeUpdateCallback(values, handle, unencoded, tap, positions) {
    currentSpeed = range.noUiSlider.get();
    if(firstSliderValue) {
        document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_right</i>';
        firstSliderValue = false;
        return;
    }
    switch(parseInt(currentSpeed)) {
        case monthBack:
            window.call_native("set_time_speed", -monthInSec);
            document.getElementById("btnPaus").innerHTML = '<i class="material-icons">pause</i>';
            document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_left</i>';
          break;
        case dayBack:
            window.call_native("set_time_speed", -dayInSec);
            document.getElementById("btnPaus").innerHTML = '<i class="material-icons">pause</i>';
            document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_left</i>';
          break;
        case hourBack:
            window.call_native("set_time_speed", -hourInSec);
            document.getElementById("btnPaus").innerHTML = '<i class="material-icons">pause</i>';
            document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_left</i>';
          break;
        case secBack:
            window.call_native("set_time_speed", secBack);
            document.getElementById("btnPaus").innerHTML = '<i class="material-icons">pause</i>';
            document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_left</i>';
            break;
        case paus:
            window.call_native("set_time_speed", 0);
            document.getElementById("btnPaus").innerHTML = '<i class="material-icons">play_arrow</i>';
         break; 
        case secForw:
            window.call_native("set_time_speed", secForw);
            document.getElementById("btnPaus").innerHTML = '<i class="material-icons">pause</i>';
            document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_right</i>';
            break;
        case hourForw:
            window.call_native("set_time_speed", hourInSec);
            document.getElementById("btnPaus").innerHTML = '<i class="material-icons">pause</i>';
            document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_right</i>';
          break;
        case dayForw:
            window.call_native("set_time_speed", dayInSec);
            document.getElementById("btnPaus").innerHTML = '<i class="material-icons">pause</i>';
            document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_right</i>';
          break;
        case monthForw:
            window.call_native("set_time_speed", monthInSec);
            document.getElementById("btnPaus").innerHTML = '<i class="material-icons">pause</i>';
            document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_right</i>';
            break;       
        default:
          // code block
      } 
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
    }
}

container.addEventListener("wheel", manuelZoomTimeline, true);

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

document.getElementById("btnPaus").onclick = togglePaus;
document.getElementById("btnDecreaseSpeed").onclick = decreaseSpeed;
document.getElementById("btnIncreaseSpeed").onclick = increaseSpeed;

document.getElementById("divContainer").addEventListener("mouseup", mouseUpCallback);