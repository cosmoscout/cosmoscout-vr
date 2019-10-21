// Format a Date to a for a human readable string DD.MM.YYYY HH:MM:SS
function formatDateReadable(date) {
    var year = date.getFullYear();
    var month = (date.getMonth() + 1).toString();
    var day = date.getDate().toString();
    var hours = date.getHours().toString();
    var minutes = date.getMinutes().toString();
    var seconds = date.getSeconds().toString();
    month = month.length > 1 ? month : '0' + month;
    day = day.length > 1 ? day : '0' + day;
    hours = hours.length > 1 ? hours : '0' + hours;
    minutes = minutes.length > 1 ? minutes : '0' + minutes;
    seconds = seconds.length > 1 ? seconds : '0' + seconds;
    return day + '.' + month + '.' + year + " " + hours + ":" + minutes + ":" + seconds;
}

// Format a Date to YYYY-MM-DD
function getFormattedDate(date) {
    var year = date.getFullYear();
    var month = (date.getMonth() + 1).toString();
    var day = date.getDate().toString();
    month = month.length > 1 ? month : '0' + month;
    day = day.length > 1 ? day : '0' + day;
    return year + '-' + month + '-' + day;
}

// Format a Date to YYYY-MM-DD HH:MM:SS
function getFormattedDateWithTime(date) {
    var retVal = getFormattedDate(date);
    var hours = date.getHours().toString();
    var minutes = date.getMinutes().toString();
    var seconds = date.getSeconds().toString();
    hours = hours.length > 1 ? hours : '0' + hours;
    minutes = minutes.length > 1 ? minutes : '0' + minutes;
    seconds = seconds.length > 1 ? seconds : '0' + seconds;
    retVal = retVal + " " + hours + ":" + minutes + ":" + seconds;
    return retVal;
}

// Format a Date to a readable format for CosmoScoutVR YYYY-MM-DD HH:MM:SS.sss
function formatDateCosmo(date) {
    var retVal = getFormattedDate(date);
    var hours = date.getHours().toString();
    var minutes = date.getMinutes().toString();
    var seconds = date.getSeconds().toString();
    var milliSec = date.getMilliseconds().toString();
    while (milliSec.length < 3) {
        milliSec = '0' + milliSec;
    }
    hours = hours.length > 1 ? hours : '0' + hours;
    minutes = minutes.length > 1 ? minutes : '0' + minutes;
    seconds = seconds.length > 1 ? seconds : '0' + seconds;
    retVal = retVal + " " + hours + ":" + minutes + ":" + seconds + "." + milliSec;
    return retVal;
}

// Convert seconds into Date
function convertSeconds(given_seconds) {
    var converted = new Object();
    converted.days = Math.floor(given_seconds / dayInSec);
    converted.hours = Math.floor((given_seconds - (converted.days * dayInSec)) / hourInSec);
    converted.minutes = Math.floor((given_seconds - (converted.days * dayInSec) - (converted.hours * hourInSec)) / minuteInSec);
    converted.seconds = Math.floor(given_seconds - (converted.days * dayInSec) - (converted.hours * hourInSec) - (converted.minutes * minuteInSec));
    converted.milliSec = Math.round((given_seconds - Math.floor(given_seconds)) * 1000);
    return converted;
}

// Increase a Date by days, hours , minutes, seconds and milliseconds
function increaseDate(date, days, hours, minutes, seconds, milliSec) {
    date.setDate(date.getDate() + days);
    date.setHours(date.getHours() + hours);
    date.setMinutes(date.getMinutes() + minutes);
    date.setSeconds(date.getSeconds() + seconds);
    date.setMilliseconds(date.getMilliseconds() + milliSec);
    return date;
}

// Decrease a Date by days, hours , minutes, seconds and milliseconds
function decreaseDate(date, days, hours, minutes, seconds, milliSec) {
    date.setDate(date.getDate() - days);
    date.setHours(date.getHours() - hours);
    date.setMinutes(date.getMinutes() - minutes);
    date.setSeconds(date.getSeconds() - seconds);
    date.setMilliseconds(date.getMilliseconds() - milliSec);
    return date;
}