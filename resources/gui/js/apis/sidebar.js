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

    tab.id        = "sidebar-tab-" + this._makeId(pluginName);
    tab.innerHTML = tab.innerHTML.replace(/%NAME%/g, pluginName)
                        .replace(/%ICON%/g, icon)
                        .replace(/%ID%/g, tab.id)
                        .replace(/%CONTENT%/g, content);

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

    tab.id        = "sidebar-settings-" + this._makeId(sectionName);
    tab.innerHTML = tab.innerHTML.replace(/%NAME%/g, sectionName)
                        .replace(/%ICON%/g, icon)
                        .replace(/%ID%/g, tab.id)
                        .replace(/%CONTENT%/g, content);

    this._settings.appendChild(tab);
  }

  /**
   * Removes a plugin tab from the sidebar
   *
   * @param pluginName {string}
   */
  removePluginTab(pluginName) {
    const id = "sidebar-tab-" + this._makeId(pluginName);
    document.getElementById(id).remove();
  }

  /**
   * Removes a settings section from the sidebar
   *
   * @param pluginName {string}
   */
  removeSettingsSection(pluginName) {
    const id = "sidebar-settings-" + this._makeId(pluginName);
    document.getElementById(id).remove();
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
