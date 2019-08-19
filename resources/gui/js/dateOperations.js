function formatDateReadable(date) {
    var retVal = date.toDateString();
    var hours = date.getHours().toString();
    var minutes = date.getMinutes().toString();
    var seconds = date.getSeconds().toString();
    hours = hours.length > 1 ? hours : '0' + hours;
    minutes = minutes.length > 1 ? minutes : '0' + minutes;
    seconds = seconds.length > 1 ? seconds : '0' + seconds;
    retVal += " " + hours + ":" + minutes + ":" + seconds;
    return retVal;
}

function getFormattedDate(date) {
    var year = date.getFullYear();
    var month = (date.getMonth() + 1).toString();
    var day = date.getDate().toString();
    month = month.length > 1 ? month : '0' + month;
    day = day.length > 1 ? day : '0' + day;
    return year + '-' + month + '-' + day;
}

function convertSeconds(given_seconds) {
    var converted = new Object();
    converted.days = Math.floor(given_seconds / dayInSec);
    converted.hours = Math.floor((given_seconds - (converted.days * dayInSec)) / hourInSec);
    converted.minutes = Math.floor((given_seconds - (converted.days * dayInSec) - (converted.hours * hourInSec)) / minuteInSec);
    converted.seconds = Math.floor(given_seconds - (converted.days * dayInSec) - (converted.hours * hourInSec) - (converted.minutes * minuteInSec));
    converted.milliSec = Math.round((given_seconds - Math.floor(given_seconds)) * 1000);
    return converted;
}

function increaseDate(date, days, hours, minutes, seconds, milliSec) {
    date.setDate( date.getDate() + days);
    date.setHours( date.getHours() + hours);
    date.setMinutes( date.getMinutes() + minutes);
    date.setSeconds( date.getSeconds() + seconds);
    date.setMilliseconds( date.getMilliseconds() + milliSec);
    return date;
}

function decreaseDate(date, days, hours, minutes, seconds, milliSec) {
    date.setDate( date.getDate() - days);
    date.setHours( date.getHours() - hours);
    date.setMinutes( date.getMinutes() - minutes);
    date.setSeconds( date.getSeconds() - seconds);
    date.setMilliseconds( date.getMilliseconds() - milliSec);
    return date;
}