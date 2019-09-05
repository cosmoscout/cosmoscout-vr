function addPluginTab(pluginName, iconName, content) {
    const id = pluginName.split(' ').join('-');
    $("#sidebar-accordion .sidebar-tab:last").before(`
    <div class="card sidebar-tab">
        <div class="card-header collapsed tab" id="heading-${id}" data-toggle="collapse" 
        data-target="#collapse-${id}" aria-expanded="false" aria-controls="collapse-${id}" type="button">
            <i class="material-icons">${iconName}</i><span class="header-name">${id}</span>
        </div>
        <div class="card-body collapse" id="collapse-${id}" aria-labelledby="heading-${id}"
             data-parent="#sidebar-accordion">
            ${content}
        </div>
    </div>
  `);
}

function addSettingsSection(sectionName, icon, content) {
    const id = sectionName.split(' ').join('-');
    $("#settings-accordion").append(`
    <div class="card settings-section">
        <div class="card-header collapsed" id="headingSettings-${id}"
             data-toggle="collapse" data-target="#collapseSettings-${id}" aria-expanded="false"
             aria-controls="collapseSettings-${id}" type="button">
            <i class="material-icons">${icon}</i>
            <span>${sectionName}</span>
            <i class="material-icons caret-icon">keyboard_arrow_left</i>
        </div>
        <div class="card-body collapse" id="collapseSettings-${id}"
             aria-labelledby="headingSettings-${id}" data-parent="#settings-accordion">
            ${content}
        </div>
    </div>
  `);
}

function hide(selector) {
    $(selector).addClass('hidden');
}

function clear_container(id) {
    $('#' + id).empty();
}

function clear_dropdown(id) {
    const element = $('#' + id);
    element.empty();
    element.selectpicker('render');
}

function set_dropdown_value(id, value) {
    $('#' + id).selectpicker('val', value);
}

function add_dropdown_value(id, opt_value, opt_text, opt_selected) {
    const selected = opt_selected ? 'selected' : '';
    const html = `<option value="${opt_value}" ${selected}>${opt_text}</option>`;

    const element = $('#' + id);
    element.append(html);
    element.selectpicker('refresh');
}

// update gui when value is set over the network -----------------------
function set_slider_value(id, value) {
    const slider = document.getElementById(id);
    slider.noUiSlider.set(value);
}

function set_slider_value(id, val1, val2) {
    const slider = document.getElementById(id);
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

function beauty_print_number(value) {
    const abs = Math.abs(value);
    if (abs >= 10000)
        return Number(value.toPrecision(2)).toExponential();
    if (abs >= 1000)
        return Number(value.toPrecision(4));
    if (abs >= 1)
        return Number(value.toPrecision(3));
    if (abs >= 0.1)
        return Number(value.toPrecision(2));
    if (abs === 0)
        return "0";

    return Number(value.toPrecision(2)).toExponential();
}

function init() {
    const dropdowns = $(".simple-value-dropdown");
    dropdowns.selectpicker();
    dropdowns.on('change', function () {
        if (this.id !== '') {
            window.call_native(this.id, this.value);
        }
    });

    $('.checklabel input').change(function () {
        window.call_native(this.id, this.checked);
    });

    $('[data-toggle="tooltip"]').tooltip({ delay: 500, placement: "auto", html: false });

    $('.radiolabel input').change(function () {
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
        range: { 'min': min, 'max': max },
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

function set_scene_luminance(value) {
    $("#scene-luminance").text(beauty_print_number(parseFloat(value)));
}

$(document).ready(function () {
    //init();

    const slider = document.getElementById("set_shadowmap_resolution");
    noUiSlider.create(slider, {
        start: 2048,
        connect: "lower",
        snap: true,
        range: { 'min': 256, '25%': 512, '50%': 1024, '75%': 2048, 'max': 4096 },
        format: wNumb({})
    });

    slider.noUiSlider.on('slide', function (values, handle, unencoded) {
        window.call_native("set_shadowmap_resolution", unencoded, 0);
    });

    // Performance
    init_slider("set_lighting_quality", 1.0, 3.0, 1.0, [2.0]);

    init_slider("set_shadowmap_cascades", 1.0, 5.0, 1.0, [3.0]);
    init_slider("set_shadowmap_bias", 0.0, 10.0, 0.01, [1.0]);
    init_slider("set_shadowmap_range", 0.0, 500.0, 1.0, [0.0, 100.0]);
    init_slider("set_shadowmap_extension", -500.0, 500.0, 10.0, [-100.0, 100.0]);
    init_slider("set_shadowmap_split_distribution", 0.5, 5.0, 0.1, [1.0]);

    // Camera
    init_slider("set_exposure", -30, 30, 0.5, [0]);
    init_slider("set_sensor_diagonal", 10.0, 70, 1, [42]);
    init_slider("set_focal_length", 10.0, 500, 1, [24]);
    init_slider("set_exposure_compensation", -10, 10, 0.5, [0]);
    init_slider("set_exposure_adaption_speed", 0, 20, 0.1, [3]);
    init_slider("set_ambient_light", 0.0, 1.0, 0.001, [0.25]);
    init_slider("set_exposure_range", -30.0, 30, 0.1, [-15, 1.5]);
    init_slider("set_glow_intensity", 0.0, 1, 0.01, [0.1]);

    $(document).on('click', '.item-create-button', function () {
        $(this).addClass('active');
        $(this).text('Cancel');
        $(document).on("click", cancel_item_creation_handler);
    });

    $('#set_enable_auto_exposure').change(function () {
        if (this.checked) {
            $("#set_exposure").addClass("unresponsive");
            $("#set_exposure_range").removeClass("unresponsive");
        } else {
            $("#set_exposure").removeClass("unresponsive");
            $("#set_exposure_range").addClass("unresponsive");
        }
    });
});
