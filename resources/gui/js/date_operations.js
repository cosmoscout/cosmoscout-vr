// Format a Date to a for a human readable string DD.MM.YYYY HH:MM:SS
function format_date_readable(date) {
    return DateOperations.formatDateReadable(date);
}

// Format a Date to YYYY-MM-DD
function get_formatted_date(date) {
    return DateOperations.getFormattedDate(date);
}

// Format a Date to YYYY-MM-DD HH:MM:SS
function get_formatted_dateWithTime(date) {
    return DateOperations.getFormattedDateWithTime(date);
}

// Format a Date to a readable format for CosmoScoutVR YYYY-MM-DD HH:MM:SS.sss
function format_date_cosmo(date) {
    return DateOperations.formatDateCosmo(date);
}

// Convert seconds into Date
function convert_seconds(given_seconds) {
    return DateOperations.convertSeconds(given_seconds);
}

// Increase a Date by days, hours , minutes, seconds and milliseconds
function increase_date(date, days, hours, minutes, seconds, milliSec) {
    return DateOperations.increaseDate(date, days, hours, minutes, seconds, milliSec);
}

// Decrease a Date by days, hours , minutes, seconds and milliseconds
function decrease_date(date, days, hours, minutes, seconds, milliSec) {
    return DateOperations.decreaseDate(date, days, hours, minutes, seconds, milliSec);
}
