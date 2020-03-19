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
   * @param tabName {string}
   * @param icon {string}
   * @param content {string}
   */
  addPluginTab(tabName, icon, content) {
    const tab = CosmoScout.gui.loadTemplateContent('sidebar-plugin-tab');
    if (tab === false) {
      console.warn('"#sidebar-plugin-tab-template" could not be loaded!');
      return;
    }

    tab.id        = "sidebar-tab-" + this._makeId(tabName);
    tab.innerHTML = tab.innerHTML.replace(/%NAME%/g, tabName)
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
   * @param tabName {string}
   */
  removePluginTab(tabName) {
    const id = "sidebar-tab-" + this._makeId(tabName);
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
   * @param tabName {string}
   * @param enabled {boolean}
   */
  setTabEnabled(tabName, enabled) {
    const id  = "sidebar-tab-" + this._makeId(tabName);
    const tab = document.getElementById(id);

    if (tab === null) {
      console.warn(`Tab with id #${id} not found!`);
      return;
    }

    if (enabled) {
      tab.classList.remove('unresponsive');
    } else {
      $(`#${id}`).collapse('hide');
      tab.classList.add('unresponsive');
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
