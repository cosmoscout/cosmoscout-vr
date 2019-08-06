function addPluginTab(pluginName, iconName, content) {
    $("#tab-list .sidebar-tab:last").before(`
    <li class="bevel sidebar-tab">
      <div class="collapsible-header tab-header waves-effect waves-light">
        <i class="material-icons">${iconName}</i>
        <span class="header-name">${pluginName}</span>
      </div>
      <div class="collapsible-body tab-body">
        ${content}
      </div>
    </li>
  `);
}

function addSettingsSection(sectionName, iconName, content) {
    $("#settings-list").append(`
    <li>
      <div class="collapsible-header settings-section waves-effect waves-light">
      <i class="material-icons">${iconName}</i><span>${sectionName}</span><i class="material-icons caret-icon">keyboard_arrow_left</i>
      </div>
      <div class="collapsible-body settings-body">
        ${content}
      </div>
    </li>
  `);
}

function hide(selector) {
    $(selector).addClass('hidden');
}

function clear_container(id) {
    $('#' + id).empty();
}

function clear_dropdown(id) {
    $('#' + id).empty();
    $('#' + id).formSelect();
}

function add_dropdown_value(id, opt_value, opt_text, opt_selected) {
    var selected = opt_selected ? 'selected' : '';
    var html = '\
                <option value="' +
        opt_value + '" ' + selected + ' >' + opt_text + '</option>';

    $('#' + id).append(html);
    $('#' + id).formSelect();
}

// update gui when value is set over the network -----------------------
function set_slider_value(id, value) {
    var slider = document.getElementById(id);
    slider.noUiSlider.set(value);
}

function set_slider_value(id, val1, val2) {
    var slider = document.getElementById(id);
    slider.noUiSlider.set([val1, val2]);
}

function set_radio_checked(id) {
    $('#' + id).prop('checked', true);
}

function set_checkbox_value(id, value) {
    $('#' + id).prop('checked', value);
}

function set_textbox_value(id, value) {
    $('.item-' + id + ' .text-input').val(value);
}

function set_dropdown_value(id, value) {
    $('#' + id).val(value);
    $('#' + id).formSelect();
}

function beauty_print_number(value) {
    var abs = Math.abs(value);
    if (abs >= 10000)
        return Number(value.toPrecision(2)).toExponential();
    if (abs >= 1000)
        return Number(value.toPrecision(4));
    if (abs >= 1)
        return Number(value.toPrecision(3));
    if (abs >= 0.1)
        return Number(value.toPrecision(2));
    if (abs == 0)
        return "0";

    return Number(value.toPrecision(2)).toExponential();
}

function set_map_data_copyright(text) {
    $("#img-data-copyright").tooltip({"html": "© " + text, "position": "top", "margin": -5});
}

function set_elevation_data_copyright(text) {
    $("#dem-data-copyright").tooltip({"html": "© " + text, "position": "bottom", "margin": -5});
}

function init() {
    // Update all with ".simple-value-dropdown" class
    $('.simple-value-dropdown').formSelect();
    $('.collapsible').collapsible();

    $('.simple-value-dropdown').on('change', function () {
        if (this.id != '') {
            window.call_native(this.id, this.value);
        }
    });

    $('.simple-value-checkbox').change(function () {
        window.call_native(this.id, this.checked);
    });

    $('.tooltipped').tooltip({'enterDelay': 500, 'margin': -8});

    $('.simple-value-radio').change(function () {
        if (this.checked) {
            window.call_native(this.id);
        }
    });
}

function init_slider(id, min, max, step, start) {
    const slider = document.getElementById(id);
    noUiSlider.create(slider, {
        start: start,
        connect: (start.length == 1 ? "lower" : true),
        step: step,
        range: {'min': min, 'max': max},
        format: {
            to: function (value) {
                return beauty_print_number(value);
            },
            from: function (value) {
                return Number(parseFloat(value));
            }
        }
    });

    slider.noUiSlider.on('slide', function (values, handle, unencoded) {
        if (Array.isArray(unencoded))
            window.call_native(id, unencoded[handle], handle);
        else
            window.call_native(id, unencoded, 0);
    });
}

$(document).ready(function () {
    //init();

    const slider = document.getElementById("set_shadowmap_resolution");
    noUiSlider.create(slider, {
        start: 2048,
        connect: "lower",
        snap: true,
        range: {'min': 256, '25%': 512, '50%': 1024, '75%': 2048, 'max': 4096},
        format: wNumb({})
    });

    slider.noUiSlider.on('slide', function (values, handle, unencoded) {
        window.call_native("set_shadowmap_resolution", unencoded, 0);
    });

    init_slider("set_ambient_light", 0.0, 1.0, 0.01, [0.5]);

    // Performance
    init_slider("set_lighting_quality", 1.0, 3.0, 1.0, [2]);

    init_slider("set_shadowmap_cascades", 1.0, 5.0, 1.0, [3]);
    init_slider("set_shadowmap_bias", 0.0, 10.0, 0.01, [1.0]);
    init_slider("set_shadowmap_range", 0.0, 500.0, 1.0, [0, 100]);
    init_slider("set_shadowmap_extension", -500.0, 500.0, 10.0, [-100, 100]);
    init_slider("set_shadowmap_split_distribution", 0.5, 5.0, 0.1, [1]);

    $(document).on('click', '.item-create-button', function () {
        $(this).addClass('active');
        $(this).text('Cancel');
        $(document).on("click", cancel_item_creation_handler);
    });
});

document.addEventListener('DOMContentLoaded', function () {
    M.AutoInit();
    let elem = document.querySelectorAll('.collapsible.expandable');
    M.Collapsible.init(elem, {
        accordion: false
    });
});