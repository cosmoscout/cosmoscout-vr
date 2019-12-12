function addPluginTab(pluginName, iconName, content) {
    CosmoScout.call('sidebar', 'addPluginTab', pluginName, iconName, content)
}

function addSettingsSection(sectionName, icon, content) {
    CosmoScout.call('sidebar', 'addSettingsSection', sectionName, icon, content);
}

function clear_container(id) {
    CosmoScout.call('sidebar', 'clearContainer', id);
}

function clear_dropdown(id) {
    CosmoScout.call('sidebar', 'clearDropdown', id);
}

function set_dropdown_value(id, value) {
    CosmoScout.call('sidebar', 'setDropdownValue', id, value);
    //$('#' + id).selectpicker('val', value);
}

function add_dropdown_value(id, opt_value, opt_text, opt_selected) {
    CosmoScout.call('sidebar', 'addDropdownValue', id, opt_value, opt_text, opt_selected);
    /*const selected = opt_selected ? 'selected' : '';
    const html = `<option value="${opt_value}" ${selected}>${opt_text}</option>`;

    const element = $('#' + id);
    element.append(html);
    element.selectpicker('refresh');*/
}

// update gui when value is set over the network -----------------------
function set_slider_value(id, value) {
    CosmoScout.call('sidebar', 'setSliderValue', id, value);

/*    const slider = document.getElementById(id);
    slider.noUiSlider.set(value);*/
}

function set_slider_value(id, val1, val2) {
    CosmoScout.call('sidebar', 'setSliderValue', id, val1, val2);
    /*const slider = document.getElementById(id);
    slider.noUiSlider.set([val1, val2]);*/
}

function set_radio_checked(id) {
    CosmoScout.call('sidebar', 'setRadioChecked', id);
//    $('#' + id).prop('checked', true);
}

function set_checkbox_value(id, value) {
    CosmoScout.call('sidebar', 'setCheckboxValue', id, value);
    //$('#' + id).prop('checked', value);
}

function set_textbox_value(id, value) {
    CosmoScout.call('sidebar', 'setTextboxValue', id, value);
//    $('.item-' + id + ' .text-input').val(value);
}

