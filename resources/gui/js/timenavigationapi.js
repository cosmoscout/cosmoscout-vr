// API calls

// Sets the timeline to the given date
function set_date(date) {
}

// Prints a notifivcatio for the time-speed and changes the slider if the time is paused
function set_time_speed(speed) {
}

// Adds a new event to the timeline
function add_item(start, end, id, content, style, description, planet, place) {
}

// Add a Button to the button bar
// @param icon The materialize icon to use
// @param tooltip Tooltip text that gets shown if the button is hovered
// @param callback Native function that gets called if the button is clicked. The function has
//                  to be registered as callback before clicking the button.
function add_button(icon, tooltip, callback) {
    var button = document.createElement("a");
    button.setAttribute('class',"btn light-glass");
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