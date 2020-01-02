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

    tab.innerHTML = this.replaceMarkers(tab.innerHTML, this._makeId(pluginName), icon, content);

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

    const html = this.replaceMarkers(tab.innerHTML, this._makeId(sectionName), icon, content);

    tab.innerHTML = html
      .replace(this.regex('SECTION'), sectionName)
      .trim();

    this._settings.appendChild(tab);
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
   * @see {addPluginTab}
   * @see {addSettingsSection}
   * @param name {string}
   * @return {string}
   * @private
   */
  _makeId(name) {
    return name.split(' ').join('-');
  }
}
