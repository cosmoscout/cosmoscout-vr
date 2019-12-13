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
}

function add_dropdown_value(id, opt_value, opt_text, opt_selected) {
    CosmoScout.call('sidebar', 'addDropdownValue', id, opt_value, opt_text, opt_selected);
}

// update gui when value is set over the network -----------------------
function set_slider_value(id, value) {
    CosmoScout.call('sidebar', 'setSliderValue', id, value);
}

function set_slider_value(id, val1, val2) {
    CosmoScout.call('sidebar', 'setSliderValue', id, val1, val2);
}

function set_radio_checked(id) {
    CosmoScout.call('sidebar', 'setRadioChecked', id);
}

function set_checkbox_value(id, value) {
    CosmoScout.call('sidebar', 'setCheckboxValue', id, value);
}

function set_textbox_value(id, value) {
    CosmoScout.call('sidebar', 'setTextboxValue', id, value);
}

