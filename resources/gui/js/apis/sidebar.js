class SidebarApi extends IApi {
  /** print_notification
   * @inheritDoc
   */
  name = 'sidebar';

  /**
   * @type {HTMLElement}
   * @private
   */
  _settings;

  /**
   * @type {HTMLElement}
   * @private
   */
  _sidebar;

  /**
   * @type {Element}
   * @private
   */
  _sidebarTab;

  /**
   * Loads all templates and needed container refs
   */
  init() {
    this._settings = document.getElementById('settings-accordion');
    this._sidebar = document.getElementById('sidebar-accordion');
    this._sidebarTab = document.getElementById('sidebar-accordion').lastElementChild;
  }

  /**
   * Add a plugin tab to the sidebar
   *
   * @param pluginName {string}
   * @param icon {string}
   * @param content {string}
   */
  addPluginTab(pluginName, icon, content) {
    const tab = CosmoScout.loadTemplateContent('sidebar-plugin-tab');
    if (tab === false) {
      return;
    }

    tab.innerHTML = this._replaceMarkers(tab.innerHTML, this._makeId(pluginName), icon, content);

    this._sidebar.insertBefore(tab, this._sidebarTab);
  }

  /**
   * Add a new section to the settings tab
   *
   * @param sectionName {string}
   * @param icon {string}
   * @param content {string}
   */
  addSettingsSection(sectionName, icon, content) {
    const tab = CosmoScout.loadTemplateContent('sidebar-settings-section');
    if (tab === false) {
      return;
    }

    const html = this._replaceMarkers(tab.innerHTML, this._makeId(sectionName), icon, content);

    tab.innerHTML = html
      .replace(this._regex('SECTION'), sectionName)
      .trim();

    this._settings.appendChild(tab);
  }

  /**
   * Adds a celestial body button to #celestial-bodies
   *
   * @param name {string}
   * @param icon {string}
   */
  addCelestialBody(name, icon) {
    const button = CosmoScout.loadTemplateContent('sidebar-celestial-body');
    if (button === false) {
      return;
    }

    button.innerHTML = button.innerHTML
      .replace(this._regex('NAME'), name)
      .replace(this._regex('ICON'), icon)
      .trim();

    button.addEventListener('click', () => {
      CosmoScout.callNative('set_celestial_body', name);
    });

    const area = document.getElementById('celestial-bodies');

    if (area === null) {
      console.error("'#celestial-bodies' not found.");
      return;
    }

    area.appendChild(button);
  }

  /**
   * @param id {string}
   */
  clearContainer(id) {
    CosmoScout.clearHtml(id);
  }

  /**
   * TODO UNUSED / Remove jQuery
   *
   * @param id {string}
   */
  clearDropdown(id) {
    CosmoScout.clearHtml(id);

    $(`#${id}`).selectpicker('render');
  }

  /**
   * csp-fly-to-locations
   *
   * @param group {string}
   * @param text {string}
   */
  addLocation(group, text) {
    let first = false;
    const tabArea = document.getElementById('location-tabs-area');

    if (tabArea.childNodes.length === 0) {
      first = true;
      tabArea.appendChild(CosmoScout.loadTemplateContent('location-tab'));
    }

    const locationsTab = document.getElementById('location-tabs');
    const tabContents = document.getElementById('nav-tabContents');

    let groupTab = document.getElementById(`nav-${group}`);

    if (groupTab === null) {
      const active = first ? 'active' : '';

      const locationTabContent = CosmoScout.loadTemplateContent('location-tab-link');
      const element = document.createElement('template');

      element.innerHTML = locationTabContent.outerHTML
        .replace(this._regex('ACTIVE'), active)
        .replace(this._regex('GROUP'), group)
        .replace(this._regex('FIRST'), first.toString())
        .trim();

      locationsTab.appendChild(element.content);

      const show = first ? 'show' : '';

      const tabContent = CosmoScout.loadTemplateContent('location-tab-pane');

      element.innerHTML = tabContent.outerHTML
        .replace(this._regex('SHOW'), show)
        .replace(this._regex('ACTIVE'), active)
        .replace(this._regex('GROUP'), group)
        .trim();

      tabContents.appendChild(element.content);

      groupTab = document.getElementById(`nav-${group}`);
    }

    const groupTabContent = CosmoScout.loadTemplateContent('location-group');

    groupTabContent.innerHTML = groupTabContent.innerHTML
      .replace(this._regex('TEXT'), text)
      .trim();

    groupTab.appendChild(groupTabContent);

    CosmoScout.initTooltips();
    CosmoScout.initDataCalls();
  }

  /**
   * Sets an elevation data copyright tooltip
   * TODO Remove jQuery
   *
   * @param copyright {string}
   */
  setElevationDataCopyright(copyright) {
    $('#img-data-copyright').tooltip({ title: `© ${copyright}`, placement: 'top' });
  }

  /**
   * Sets a map data copyright tooltip
   * TODO Remove jQuery
   *
   * @param copyright {string}
   */
  setMapDataCopyright(copyright) {
    $('#dem-data-copyright').tooltip({ title: `© ${copyright}`, placement: 'bottom' });
  }

  /**
   * Adds an option to a dropdown
   * TODO remove jQuery
   *
   * @param id {string} DropDown ID
   * @param value {string|number} Option value
   * @param text {string} Option text
   * @param selected {boolean} Selected flag
   */
  addDropdownValue(id, value, text, selected = false) {
    const dropdown = document.getElementById(id);
    const option = document.createElement('option');

    option.value = value;
    option.selected = selected === true;
    option.text = text;

    if (dropdown !== null) {
      dropdown.appendChild(option);

      $(`#${id}`).selectpicker('refresh');
    } else {
      console.warn(`Dropdown '${id} 'not found`);
    }
  }

  /**
   * TODO UNUSED
   *
   * @param id {string}
   * @param value {string|number}
   */
  setDropdownValue(id, value) {
    $(`#${id}`).selectpicker('val', value);
  }

  /**
   * TODO
   * @param name {string}
   * @param icon {string}
   */
  addMeasurementTool(name, icon) {
    const tool = CosmoScout.loadTemplateContent('measurement-tools');

    tool.innerHTML = tool.innerHTML
      .replace(this._regex('CONTENT'), name)
      .replace(this._regex('ICON'), icon)
      .trim();

    tool.addEventListener('click', () => {
      CosmoScout.callNative('set_celestial_body', name);
    });

    tool.addEventListener('change', function () {
      document.querySelectorAll('#measurement-tools .radio-button').forEach((node) => {
        if (node.id !== `set_tool_${icon}`) {
          node.checked = false;
        }
      });

      if (this.checked) {
        CosmoScout.callNative('set_measurement_tool', name);
      } else {
        CosmoScout.callNative('set_measurement_tool', 'none');
      }
    });
  }

  /**
   * Deselect all measurement tools
   */
  deselectMeasurementTool() {
    document.querySelectorAll('#measurement-tools .radio-button').forEach((node) => {
      node.checked = false;
    });
  }

  /**
   * @param file {string}
   * @param time {string|number}
   */
  addSharad(file, time) {
    const sharadList = document.getElementById('list-sharad');
    const sharad = CosmoScout.loadTemplateContent('sharad');

    sharad.innerHTML = sharad.innerHTML
      .replace(this._regex('FILE'), file)
      .replace(this._regex('TIME'), time)
      .trim();

    sharad.classList.add(`item-${file}`);

    sharadList.appendChild(sharad);

    CosmoScout.initDataCalls();
  }

  /**
   * TODO UNUSED
   * @param id {string}
   */
  setRadioChecked(id) {
    this.setCheckboxValue(id, true);
  }

  /**
   * TODO UNUSED
   * @param id {string}
   * @param value {boolean}
   */
  setCheckboxValue(id, value) {
    const element = document.getElementById(id);

    if (element !== null) {
      element.checked = value === true;
    }
  }

  /**
   * TODO UNUSED
   * @param id {string}
   * @param value {string}
   */
  setTextboxValue(id, value) {
    const element = document.querySelector(`.item-${id} .text-input`);

    if (element !== null) {
      element.value = value;
    }
  }

  /**
   * @see {addPluginTab}
   * @see {addSettingsSection}
   * @param name {string}
   * @return {string}
   * @private
   */
  _makeId(name) {
    return name.split(' ').join('-');
  }

  /**
   * Replace common template markers with content
   *
   * @param html {string}
   * @param id {string}
   * @param icon {string}
   * @param content {string}
   * @return {string}
   * @private
   */
  _replaceMarkers(html, id, icon, content) {
    return html
      .replace(this._regex('ID'), id)
      .replace(this._regex('CONTENT'), content)
      .replace(this._regex('ICON'), icon)
      .trim();
  }

  /**
   * Creates a search global Regex Object of %matcher%
   *
   * @param matcher {string}
   * @return {RegExp}
   * @private
   */
  _regex(matcher) {
    return new RegExp(`\%${matcher}\%`, 'g');
  }
}
