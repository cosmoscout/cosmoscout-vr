/* global IApi, CosmoScout */

/**
 * Sidebar Api
 */
class SidebarApi extends IApi {
  /**
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
    this._settings   = document.getElementById('settings-accordion');
    this._sidebar    = document.getElementById('sidebar-accordion');
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
    const tab = CosmoScout.gui.loadTemplateContent('sidebar-plugin-tab');
    if (tab === false) {
      console.warn('"#sidebar-plugin-tab-template" could not be loaded!');
      return;
    }

    const id = this._makeId(pluginName);

    tab.innerHTML = this._replaceMarkers(tab.innerHTML, id, icon, content);

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
    const tab = CosmoScout.gui.loadTemplateContent('sidebar-settings-section');
    if (tab === false) {
      console.warn('"#sidebar-settings-section-template" could not be loaded!');
      return;
    }

    const html = this._replaceMarkers(tab.innerHTML, this._makeId(sectionName), icon, content);

    tab.innerHTML = html.replace(/%SECTION%/g, sectionName).trim();

    this._settings.appendChild(tab);
  }

  /**
   * Enables or disables a plugin tab.
   * Disabled tabs will be collapsed if open.
   *
   * @param collapseId {string}
   * @param enabled {boolean}
   */
  setTabEnabled(collapseId, enabled) {
    const tab = document.getElementById(collapseId);

    if (tab === null) {
      console.warn(`Tab with id #${collapseId} not found!`);
      return;
    }

    // Add unresponsive class to parent element
    // Or tab if no parent is present
    // We assume tabs are contained in .sidebar-tab elements
    let parent = tab.parentElement;
    if (parent === null) {
      parent = tab;
    }

    if (enabled) {
      parent.classList.remove('unresponsive');
    } else {
      $(`#${collapseId}`).collapse('hide');
      parent.classList.add('unresponsive');
    }
  }

  setAverageSceneLuminance(value) {
    $("#average-scene-luminance").text(CosmoScout.utils.beautifyNumber(parseFloat(value)));
  }

  setMaximumSceneLuminance(value) {
    $("#maximum-scene-luminance").text(CosmoScout.utils.beautifyNumber(parseFloat(value)));
  }

  /**
   * Replace common template markers with content.
   *
   * @param html {string} HTML with %MARKER% markers
   * @param id {string} Id marker replacement
   * @param icon {string} Icon marker replacement
   * @param content {string} Content marker replacement
   * @return {string} replaced html
   * @protected
   */
  _replaceMarkers(html, id, icon, content) {
    return html.replace(/%ID%/g, id).replace(/%CONTENT%/g, content).replace(/%ICON%/g, icon).trim();
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
