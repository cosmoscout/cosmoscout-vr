// API calls

function set_date(date) {
}

function set_time_speed(speed) {
}

function add_item(start, end, id, content, style, description, planet, place) {
}

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