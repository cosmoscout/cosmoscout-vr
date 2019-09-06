let monthInSec = 2628000;
let dayInSec = 86400;
let hourInSec =  3600;
let minuteInSec = 60;

let minuteInHours = 0.01666666666666;
let dayInHours = 24;

let timeId = 'custom';

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

var parHolder = new Object();

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
    editable: {
        add: true,         // add new items by double tapping
        updateTime: true,  // drag items horizontally
        updateGroup: false, // drag items from one group to another
        remove: false,       // delete an item by tapping the delete button top right
        overrideItems: false  // allow these options to override item.editable
    },
    onAdd: onAddCallback,
    onUpdate: onUpdateCallback
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
        '82%' : secForw,
        '88%' : hourForw,
        '94%' : dayForw,
        'max' : monthForw,
    },
    snap: true,
    start: secForw
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
timeline.on('itemover', itemoverCallback);
timeline.on('itemout', itemoutCallback);

//create overview timeline
overviewTimeLine = new vis.Timeline(overviewContainer, items, overviewOptions);
overviewTimeLine.addCustomTime(timeline.getWindow().end, rightTimeId);
overviewTimeLine.addCustomTime(timeline.getWindow().start, leftTimeId);
overviewTimeLine.on('select', onSelect);
overviewTimeLine.on('click', onOverviewClick);
overviewTimeLine.on('changed', overviewChangeCallback);
overviewTimeLine.on('mouseDown', overviewMouseDownCallback);
overviewTimeLine.on('rangechange', overviewRangechangeCallback);
initialOverviewWindow(new Date(1950,1), new Date(2030, 12));

document.getElementById("dateLabel").innerText = formatDateReadable(centerTime);

moveWindow(secSpeed);

function redrawTooltip(event) {
    return new Promise(resolve => {
    if(tooltipVisible) {
        var eventRect = event.getBoundingClientRect();
        document.getElementById("customTooltip").style.top = eventRect.bottom + 'px';
        document.getElementById("customTooltip").style.left = eventRect.left + 'px';
    }

    setTimeout(function() {
        resolve(10);
        redrawTooltip(event);
        }, redrawRate);
    });
  }

async function startRedrawTooltip(event) {
    await redrawTooltip(event);
}

var hoveredItem;
var tooltipVisible = false;
function itemoverCallback(properties) {
    document.getElementById("customTooltip").style.display = "block";
    tooltipVisible = true;
    for(var item in items._data) {
        if(items._data[item].id == properties.item) {
            document.getElementById("itemContent").innerHTML = items._data[item].content;
            document.getElementById("itemDescription").innerHTML = items._data[item].description;
            document.getElementById("itemLocation").innerHTML = items._data[item].planet + " " +  items._data[item].place;
            hoveredItem = items._data[item];
        }
    }
    var events = document.getElementsByClassName(properties.item);
    var event;
    for(var i=0; i<events.length; i++) {
        if($(events[i]).hasClass("event")) {
            event = events[i];
        }
    }
    var eventRect = event.getBoundingClientRect();
    document.getElementById("customTooltip").style.top = eventRect.bottom + 'px';
    document.getElementById("customTooltip").style.left = eventRect.left + 'px';
    if(currentSpeed != paus) {
        startRedrawTooltip(event);
    }
}

function itemoutCallback(properties) {
    if(properties.event.toElement.className != "custom-tooltip-container") {
        document.getElementById("customTooltip").style.display = "none";
        tooltipVisible = false;
    }
}

function travelToItemLocation() {
    geo_code(hoveredItem.planet, hoveredItem.place);
}

function leaveCustomTooltip(event) {
    document.getElementById("customTooltip").style.display = "none";
    tooltipVisible = false;
    if(event.toElement == null) {
        mouseLeaveTimenavigation(event);
    }
}

function saveItems() {
    var data = items.get({
        type: {
          start: 'ISODate',
          end: 'ISODate'
        }
    });
}

function closeForm() {
    parHolder.callback(null); // cancel item creation
    document.getElementById("myForm").style.display = "none";
    timeline.setOptions(editingDoneOpt);
}


function applyEvent() {  
    if (document.getElementById("eventName").value != ""
    && document.getElementById("eventStartDate").value != "") {
        parHolder.item.style = "background-color: " + document.getElementById("eventColor").value;
        parHolder.item.content = document.getElementById("eventName").value;
        parHolder.item.start = new Date(document.getElementById("eventStartDate").value);
        parHolder.item.className = 'tooltipped';
        if(document.getElementById("eventEndDate").value != "") {
            parHolder.item.end = new Date(document.getElementById("eventEndDate").value);
            
        }
        parHolder.callback(parHolder.item); // send back adjusted new item
        document.getElementById("myForm").style.display = "none";
        timeline.setOptions(editingDoneOpt);
        tooltip(parHolder.item.content);
        saveItems();
    }
}

function onUpdateCallback(item, callback) {
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
    setPaus();
}

function onAddCallback(item, callback) {
    timeline.setOptions(whileEditingOpt);
    document.getElementById("headlineForm").innerText = "Add";
    document.getElementById("eventName").value = "";
    document.getElementById("myForm").style.display = "block";
    document.getElementById("eventStartDate").value = getFormattedDate(item.start);
    document.getElementById("eventEndDate").value = "";
    parHolder.item = item;
    parHolder.callback = callback;
    setPaus();
}


function setTimelineRange(min, max) {
    var rangeOpt = {
        min: min,
        max: max
    }
    timeline.setOptions(rangeOpt);
}

function mouseDownCallback() {
    timeline.setOptions(pausOpt);
    mouseOnTimelineDown = true;
    lastPlayValue = currentSpeed;
    click = true;
    mouseDownLeftTime = timeline.getWindow().start;
}

function mouseUpCallback() {
    if(mouseOnTimelineDown && lastPlayValue != paus) {
        range.noUiSlider.set(parseInt(lastPlayValue));
    }
    mouseOnTimelineDown = false;
    inRangeChange = false;
}

function overviewMouseDownCallback() {
    click = true;
}

function overviewRangechangeCallback() {
    click = false;
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
        if(currentSpeed != paus) {
            setPaus();
        }
        click = false;
        inRangeChange = true;
        var dif = properties.start.getTime() - mouseDownLeftTime.getTime();
        var secondsDif = dif/ 1000;
        var hoursDif = secondsDif / 60 / 60;
        var step = convertSeconds(secondsDif);
        var date = new Date(centerTime.getTime());
        date = increaseDate(date, step.days, step.hours, step.minutes, step.seconds, step.milliSec);
        set_date_local(date);
        mouseDownLeftTime = new Date(properties.start.getTime());
        window.call_native("add_hours_without_animation", hoursDif);
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
            var dif = items._data[item].start.getTime() - centerTime.getTime();
            var hoursDif = dif / 1000 / 60 / 60;
            window.call_native("add_hours", hoursDif);
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

function redrawSnipped() {
    return new Promise(resolve => {
    switch(parseInt(currentSpeed)) {
        case paus:
            timeline.setOptions(pausOpt);
    }

    setTimeout(function() {
        resolve(10);
        redrawSnipped();
        }, redrawRate);
    });
  }

  async function startRedrawSnipped() {
    await redrawSnipped();
  }

function tooltip(content) {
    var events = document.getElementsByClassName('event');
    for(var i=0; i<events.length; i++) {
        events[i].setAttribute('data-toggle', 'tooltip');
        events[i].setAttribute('title', content);
    }
    $('[data-toggle="tooltip"]').tooltip({ delay: 500, placement: "top", html: false });
}

function add_item(start, end, id, content, style, description, planet, place) {
    var data = new Object();
    data.start = new Date(start);
    data.id = id;
    if(end != "") {
        data.end = end;
    }
    if(style != "") {
        data.style = style;
    }
    data.planet = planet;
    data.description = description;
    data.place = place;
    data.content = content;
    data.className = 'event ' + id;
    items.update(data);
    tooltip(content);
}

function generalOnClick(properties) {
    if(properties.what != "item" && properties.time != null) {
        var dif = properties.time.getTime() - centerTime.getTime();
        var hoursDif = dif / 1000 / 60 / 60;
        window.call_native("add_hours", hoursDif);
    }
}

function onClickCallback(properties) {
    if(click) {     
        generalOnClick(properties);
    }
}

function onOverviewClick (properties){
    if(click) {     
        generalOnClick(properties);
    }
}


function setTimeToDate(date) {
    date.setHours(middleOfDay);
    window.call_native("set_date", formatDateCosmo(new Date(date.getTime())));
    var startDate = new Date(date.getTime());
    var endDate = new Date(date.getTime());
    startDate.setHours(startOfDay);
    endDate.setHours(endOfDay);
    setPaus();
    timeline.setWindow(startDate, endDate, animationFalse);
    setOverviewTimes();
}

function plusOneMinute() {
    window.call_native("add_hours", minuteInHours);
}

function minusOneMinute() {
    window.call_native("add_hours", -minuteInHours);
}

function plusOneHour() {
    window.call_native("add_hours", 1);
}
function minusOneHour() {
    window.call_native("add_hours", -1);
}

function plusOneDay() {
    window.call_native("add_hours", dayInHours);
    
}
function minusOneDay() {
    window.call_native("add_hours", -dayInHours);
}

function plusOneMonth() {
    var date = new Date(centerTime.getTime());
    centerTime.setMonth(centerTime.getMonth() + 1);
    var dif = centerTime.getTime() - date.getTime();
    var hoursDif = dif / 1000 / 60 / 60;
    window.call_native("add_hours", hoursDif);
}
function minusOneMonth() {
    var date = new Date(centerTime.getTime());
    centerTime.setMonth(centerTime.getMonth() - 1);
    var dif = centerTime.getTime() - date.getTime();
    var hoursDif = dif / 1000 / 60 / 60;
    window.call_native("add_hours", hoursDif);
}

function plusOneYear() {
    var date = new Date(centerTime.getTime());
    centerTime.setFullYear(centerTime.getFullYear() + 1);
    var dif = centerTime.getTime() - date.getTime();
    var hoursDif = dif / 1000 / 60 / 60;
    window.call_native("add_hours", hoursDif);
}
function minusOneYear() {
    var date = new Date(centerTime.getTime());
    centerTime.setFullYear(centerTime.getFullYear() - 1);
    var dif = centerTime.getTime() - date.getTime();
    var hoursDif = dif / 1000 / 60 / 60;
    window.call_native("add_hours", hoursDif);
}



function moveWindow(seconds) {
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

  var time_speed   = 1.0;
  let hour_speed   = 3600;
  let day_speed    = 86400;
  let month_speed  = 2628000;
  function set_time_speed(speed) {
      $("#play-pause-icon").text("pause");
      if (speed == 0.0) {
          $("#play-pause-icon").text("play_arrow");
          setPaus();
          window.call_native("print_notification", "Pause", "Time is paused.", "pause");
      } else if (speed == 1.0) {
          window.call_native("print_notification", "Speed: Realtime", "Time runs in realtime.", "play_arrow");
      } else if (speed == hour_speed) {
          window.call_native("print_notification", "Speed: Hour/s", "Time runs at one hour per second.", "fast_forward");
      }else if (speed == day_speed) {
          window.call_native("print_notification", "Speed: Day/s", "Time runs at one day per second.", "fast_forward");
      }else if (speed == month_speed) {
          window.call_native("print_notification", "Speed: Month/s", "Time runs at one month per second.", "fast_forward");
      }else if (speed > time_speed) {
          window.call_native("print_notification", "Speed: " + speed + "x", "Time speed increased.", "fast_forward");
      } else if (speed < time_speed) {
          window.call_native("print_notification", "Speed: " + speed + "x", "Time speed decreased.", "fast_rewind");
      }

      time_speed = speed;
  }

function setPaus() {
    currentSpeed = paus;
    window.call_native("set_time_speed", 0);
    document.getElementById("btnPaus").innerHTML = '<i class="material-icons">play_arrow</i>';
    document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">pause</i>';
    timeline.setOptions(pausOpt);
    timelineZoomBlocked = false;
    startRedrawSnipped();
}

function togglePaus() {
    if(currentSpeed != paus) {
        lastPlayValue = range.noUiSlider.get();;
        setPaus();
    } else {
        if(lastPlayValue == paus) {
            lastPlayValue = secForw;
        }
        range.noUiSlider.set(parseInt(lastPlayValue));
    }
}

function decreaseSpeed() {
    if(currentSpeed == paus) {
        togglePaus();
    } else {
        range.noUiSlider.set(currentSpeed-1);
    }
}

function increaseSpeed() {
    if(currentSpeed == paus) {
        togglePaus();
    }else {
       if(currentSpeed == secBack) {
            range.noUiSlider.set(secForw);
       }else {
            range.noUiSlider.set(currentSpeed-(-1));
       }
    }
}

function rangeUpdateCallback() {
    currentSpeed = range.noUiSlider.get();
    if(firstSliderValue) {
        document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_right</i>';
        firstSliderValue = false;
        return;
    }

    document.getElementById("btnPaus").innerHTML = '<i class="material-icons">pause</i>';
    timeline.setOptions(playingOpt);
    timelineZoomBlocked = true;
    if(parseInt(currentSpeed) < paus) {
        document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_left</i>';
    } else {
        document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_right</i>';
    }

    switch(parseInt(currentSpeed)) {
        case monthBack:
            window.call_native("set_time_speed", -monthInSec);
            moveWindow(monthSpeed);
          break;
        case dayBack:
            window.call_native("set_time_speed", -dayInSec);
            moveWindow(daySpeed);
          break;
        case hourBack:
            window.call_native("set_time_speed", -hourInSec);
            moveWindow(hourSpeed);
          break;
        case secBack:
            window.call_native("set_time_speed", secBack);
            moveWindow(secSpeed);
            break;
        case secForw:
            window.call_native("set_time_speed", secForw);
            moveWindow(secSpeed);
            break;
        case hourForw:
            window.call_native("set_time_speed", hourInSec);
            moveWindow(hourSpeed);
          break;
        case dayForw:
            window.call_native("set_time_speed", dayInSec);
            moveWindow(daySpeed);
          break;
        case monthForw:
            window.call_native("set_time_speed", monthInSec);
            moveWindow(monthSpeed);
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
        rangeUpdateCallback();
    }
}

function scrollOnYear(event) {
    if(event.deltaY < 0) {
        plusOneYear();
    } else {
        minusOneYear();
    }
}

function scrollOnMonth(event) {
    if(event.deltaY < 0) {
        plusOneMonth();
    } else {
        minusOneMonth();
    }
}

function scrollOnDay(event) {
    if(event.deltaY < 0) {
        window.call_native("add_hours_without_animation", dayInHours);
    } else {
        window.call_native("add_hours_without_animation", -dayInHours);
    }
}

function scrollOnHour(event) {
    if(event.deltaY < 0) {
        window.call_native("add_hours_without_animation", 1);
    } else {
        window.call_native("add_hours_without_animation", -1);
    }
}

function scrollOnMinute(event) {
    if(event.deltaY < 0) {
        window.call_native("add_hours_without_animation", minuteInHours);
    } else {
        window.call_native("add_hours_without_animation", -minuteInHours);
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

document.getElementById("itemLocation").onclick = travelToItemLocation;

document.getElementById("divContainer").addEventListener("mouseup", mouseUpCallback);

document.getElementsByClassName('range-label')[0].addEventListener('mousedown', rangeUpdateCallback);

document.getElementById("btnDecreaseYear").addEventListener("wheel", scrollOnYear);
document.getElementById("btnDecreaseMonth").addEventListener("wheel", scrollOnMonth);
document.getElementById("btnDecreaseDay").addEventListener("wheel", scrollOnDay);
document.getElementById("btnDecreaseHour").addEventListener("wheel", scrollOnHour);
document.getElementById("btnDecreaseMinute").addEventListener("wheel", scrollOnMinute);

document.getElementById("btnIncreaseYear").addEventListener("wheel", scrollOnYear);
document.getElementById("btnIncreaseMonth").addEventListener("wheel", scrollOnMonth);
document.getElementById("btnIncreaseDay").addEventListener("wheel", scrollOnDay);
document.getElementById("btnIncreaseHour").addEventListener("wheel", scrollOnHour);
document.getElementById("btnIncreaseMinute").addEventListener("wheel", scrollOnMinute);

document.getElementById("btnCancel").onclick = closeForm;
document.getElementById("btnApply").onclick = applyEvent;

document.getElementById("customTooltip").onmouseleave = leaveCustomTooltip;
