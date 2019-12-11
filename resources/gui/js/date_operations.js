// Format a Date to a for a human readable string DD.MM.YYYY HH:MM:SS
function format_date_readable(date) {
    let year = date.getFullYear();
    let month = (date.getMonth() + 1).toString();
    let day = date.getDate().toString();
    let hours = date.getHours().toString();
    let minutes = date.getMinutes().toString();
    let seconds = date.getSeconds().toString();
    month = month.length > 1 ? month : '0' + month;
    day = day.length > 1 ? day : '0' + day;
    hours = hours.length > 1 ? hours : '0' + hours;
    minutes = minutes.length > 1 ? minutes : '0' + minutes;
    seconds = seconds.length > 1 ? seconds : '0' + seconds;
    return day + '.' + month + '.' + year + " " + hours + ":" + minutes + ":" + seconds;
}

// Format a Date to YYYY-MM-DD
function get_formatted_date(date) {
    let year = date.getFullYear();
    let month = (date.getMonth() + 1).toString();
    let day = date.getDate().toString();
    month = month.length > 1 ? month : '0' + month;
    day = day.length > 1 ? day : '0' + day;
    return year + '-' + month + '-' + day;
}

// Format a Date to YYYY-MM-DD HH:MM:SS
function get_formatted_dateWithTime(date) {
    let retVal = get_formatted_date(date);
    let hours = date.getHours().toString();
    let minutes = date.getMinutes().toString();
    let seconds = date.getSeconds().toString();
    hours = hours.length > 1 ? hours : '0' + hours;
    minutes = minutes.length > 1 ? minutes : '0' + minutes;
    seconds = seconds.length > 1 ? seconds : '0' + seconds;
    retVal = retVal + " " + hours + ":" + minutes + ":" + seconds;
    return retVal;
}

// Format a Date to a readable format for CosmoScoutVR YYYY-MM-DD HH:MM:SS.sss
function format_date_cosmo(date) {
    let retVal = get_formatted_date(date);
    let hours = date.getHours().toString();
    let minutes = date.getMinutes().toString();
    let seconds = date.getSeconds().toString();
    let milliSec = date.getMilliseconds().toString();
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
function convert_seconds(given_seconds) {
    let converted = {};
    converted.days = Math.floor(given_seconds / dayInSec);
    converted.hours = Math.floor((given_seconds - (converted.days * dayInSec)) / hourInSec);
    converted.minutes = Math.floor((given_seconds - (converted.days * dayInSec) - (converted.hours * hourInSec)) / minuteInSec);
    converted.seconds = Math.floor(given_seconds - (converted.days * dayInSec) - (converted.hours * hourInSec) - (converted.minutes * minuteInSec));
    converted.milliSec = Math.round((given_seconds - Math.floor(given_seconds)) * 1000);
    return converted;
}

// Increase a Date by days, hours , minutes, seconds and milliseconds
function increase_date(date, days, hours, minutes, seconds, milliSec) {
    date.setDate(date.getDate() + days);
    date.setHours(date.getHours() + hours);
    date.setMinutes(date.getMinutes() + minutes);
    date.setSeconds(date.getSeconds() + seconds);
    date.setMilliseconds(date.getMilliseconds() + milliSec);
    return date;
}

// Decrease a Date by days, hours , minutes, seconds and milliseconds
function decrease_date(date, days, hours, minutes, seconds, milliSec) {
    date.setDate(date.getDate() - days);
    date.setHours(date.getHours() - hours);
    date.setMinutes(date.getMinutes() - minutes);
    date.setSeconds(date.getSeconds() - seconds);
    date.setMilliseconds(date.getMilliseconds() - milliSec);
    return date;
}