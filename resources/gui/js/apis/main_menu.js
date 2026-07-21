////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

/**
 * API class for managing the main menu interface.
 * Handles navigation, plugin management, and scene save/load operations.
 * @extends IApi
 */
class MainMenuApi extends IApi {
  name = 'mainMenu';

  /**
   * Initializes the main menu API.
   * Sets up DOM references, event listeners, and menu configurations.
   */
  init() {
    this._currentMenu = 'root';
    this._history     = [];

    this._menuElement       = document.querySelector('#cs-main-menu');
    this._menuBodyElement   = this._menuElement.querySelector('#main-menu-body');
    this._backButtonElement = this._menuElement.querySelector('#main-menu-back-button');

    this._menus = {
      root: {
        title: 'Main Menu',
        icon: 'menu',
        items: [
          {
            label: 'Manage Plugins',
            icon: 'extension',
            target: 'plugins',
          },
          {
            label: 'Save',
            icon: 'save',
            target: 'save',
          },
          {
            label: 'Load',
            icon: 'folder_open',
            target: 'load',
          },
          {
            label: 'Exit to Desktop',
            icon: 'exit_to_app',
            action: () => CosmoScout.callbacks.core.exit(),
          },
        ],
      },

      plugins: {
        title: 'Plugins',
        icon: 'extension',
        render: () => this._renderPlugins(),
      },
      save: {
        title: 'Save',
        icon: 'save',
        render: () => this._renderSaveMenu(),
      },
      load: {
        title: 'Load',
        icon: 'folder_open',
        render: () => this._renderLoadMenu(),
      },
    };

    this._backButtonElement.addEventListener('click', () => this.back());

    this._menuBodyElement.addEventListener('click', (event) => this._handleBodyClick(event));

    this._menuElement.addEventListener('mousedown', (event) => {
      if (event.target === event.currentTarget) {
        this.close();
      }
    });

    this.render();
  }

  /**
   * Navigates to a specified menu.
   * @param {string} menuId - The ID of the menu to navigate to.
   */
  navigate(menuId) {
    if (!this._menus[menuId]) {
      console.warn(`Unknown menu: ${menuId}`);
      return;
    }

    this._history.push(this._currentMenu);
    this._currentMenu = menuId;
    this.render();
  }

  /**
   * Returns to the previous menu in the navigation history.
   */
  back() {
    if (this._history.length === 0) {
      return;
    }

    this._currentMenu = this._history.pop();
    this.render();
  }

  /**
   * Handles click events on the menu body.
   * @param {MouseEvent} event - The click event.
   */
  _handleBodyClick(event) {
    const menu = this._menus[this._currentMenu];

    const menuItem = event.target.closest('.main-menu-item');
    if (menuItem && menu.items) {
      const item = menu.items[Number(menuItem.dataset.index)];

      if (!item) {
        return;
      }

      if (item.target) {
        this.navigate(item.target);
      } else if (item.action) {
        item.action();
      }

      return;
    }

    const pluginAction = event.target.closest('[data-plugin-action]');
    if (pluginAction) {
      const pluginName = pluginAction.dataset.pluginName;
      const action     = pluginAction.dataset.pluginAction;

      this._runPluginAction(pluginName, action);
    }
  }

  /**
   * Executes a plugin action.
   * @param {string} pluginName - The name of the plugin.
   * @param {string} action - The action to perform (load, unload, reload).
   */
  _runPluginAction(pluginName, action) {
    if (!pluginName || !action) {
      return;
    }

    let callback = null;

    if (action === 'reload') {
      CosmoScout.callbacks.core.reloadPlugin(pluginName);
      callback = this._waitForPluginStateChange(pluginName);
    } else if (action === 'unload') {
      CosmoScout.callbacks.core.unloadPlugin(pluginName);
      callback = this._waitForPluginStateChange(pluginName);
    } else if (action === 'load') {
      CosmoScout.callbacks.core.loadPlugin(pluginName);
      callback = this._waitForPluginStateChange(pluginName);
    }

    if (!callback) {
      console.warn(`Unknown plugin action: ${action}`);
      return;
    }

    callback.then(() => this.render());
  }

  /**
   * Waits for a plugin to actually load or unload.
   * @param {string} pluginName - The name of the plugin to wait for.
   * @returns {Promise<void>} Promise that resolves when the plugin state has changed.
   */
  _waitForPluginStateChange(pluginName) {
    return new Promise((resolve) => {
      let initialCheckDone = false;
      let wasActive        = false;

      const checkPluginState = () => {
        CosmoScout.callbacks.core.getPlugins().then((plugins) => {
          const isActive = plugins[pluginName] === true;

          if (!initialCheckDone) {
            wasActive        = isActive;
            initialCheckDone = true;
            requestAnimationFrame(checkPluginState);
            return;
          }

          if (isActive !== wasActive) {
            resolve();
          } else {
            requestAnimationFrame(checkPluginState);
          }
        });
      };
      checkPluginState();
    });
  }

  /**
   * Renders menu items into the body element.
   * @param {Object} menu - The menu configuration object.
   * @param {HTMLElement} body - The container element to append items to.
   */
  _renderMenuItems(menu, body) {
    menu.items.forEach((item, index) => {
      const element = CosmoScout.gui.loadTemplateContent('main-menu-item-template');
      if (element === false) {
        return;
      }

      element.dataset.index = index;

      const icon = element.querySelector('.main-menu-item-icon');
      if (item.icon) {
        icon.textContent = item.icon;
      } else {
        icon.remove();
      }

      element.querySelector('.main-menu-item-label').textContent = item.label;
      body.appendChild(element);
    });
  }

  /**
   * Renders the plugins management menu.
   * Displays a list of available plugins with load/unload/reload actions.
   */
  _renderPlugins() {
    this._menuBodyElement.innerHTML = '';

    const pluginManager = CosmoScout.gui.loadTemplateContent('main-menu-plugin-template');
    if (pluginManager === false) {
      return;
    }

    this._menuBodyElement.appendChild(pluginManager);

    CosmoScout.callbacks.core.getPlugins().then((plugins) => {
      if (this._currentMenu !== 'plugins') {
        return;
      }

      const pluginList     = pluginManager.querySelector('.plugin-manager-items');
      pluginList.innerHTML = '';

      Object.entries(plugins).forEach(([name, active]) => {
        const templateId = active ? 'main-menu-plugin-item-active-template'
                                  : 'main-menu-plugin-item-inactive-template';
        const pluginItem = CosmoScout.gui.loadTemplateContent(templateId);

        if (pluginItem === false) {
          return;
        }

        pluginItem.dataset.pluginName                                     = name;
        pluginItem.querySelector('.plugin-manager-item-name').textContent = name;

        pluginItem.querySelectorAll('[data-plugin-action]')
            .forEach((actionButton) => { actionButton.dataset.pluginName = name; });

        pluginList.appendChild(pluginItem);
      });
    });
  }

  /**
   * Renders the save scene menu.
   * Allows users to create new saves or overwrite existing ones.
   */
  _renderSaveMenu() {
    this._menuBodyElement.innerHTML = '';

    const saveMenu = CosmoScout.gui.loadTemplateContent('main-menu-save-template');
    if (saveMenu === false) {
      return;
    }

    this._menuBodyElement.appendChild(saveMenu);

    const newFileSection = CosmoScout.gui.loadTemplateContent('main-menu-save-new-template');
    if (newFileSection === false) {
      return;
    }
    saveMenu.appendChild(newFileSection);

    const existingSavesSection = CosmoScout.gui.loadTemplateContent('main-menu-save-existing-template');
    if (existingSavesSection === false) {
      return;
    }
    saveMenu.appendChild(existingSavesSection);

    this._loadSaveFiles().then((saveFiles) => {
      const saveList = existingSavesSection.querySelector('.save-list');

      if (saveFiles.length === 0) {
        const emptyTemplate = CosmoScout.gui.loadTemplateContent('main-menu-save-empty-template');
        if (emptyTemplate !== false) {
          saveList.appendChild(emptyTemplate);
        }
      }

      saveFiles.forEach((saveFile, index) => {
        const saveItem = CosmoScout.gui.loadTemplateContent('main-menu-save-item-template');
        if (saveItem === false) {
          return;
        }

        saveItem.dataset.saveIndex                            = index;
        saveItem.querySelector('.save-item-name').textContent = saveFile.name;
        saveItem.querySelector('.save-item-date').textContent = saveFile.date;

        saveList.appendChild(saveItem);
      });

      document.querySelectorAll('[data-save-action="overwrite"]').forEach((button) => {
        button.addEventListener('click', (event) => {
          const saveIndex = event.currentTarget.dataset.saveIndex;
          this._handleOverwriteSave(saveIndex);
        });
      });

      document.querySelector('#save-new-button')
          .addEventListener('click', () => this._handleNewSave());
    });
  }

  /**
   * Loads and formats the list of save files.
   * @returns {Promise<Array<Object>>} Promise resolving to an array of save file objects with name
   *     and date.
   */
  _loadSaveFiles() {
    return CosmoScout.callbacks.core.getSaveFiles().then((saveFiles) => {
      return saveFiles.map((file) => {
        const name     = file.name || 'Untitled';
        const basename = name.split(/[\\/]/).pop();
        return {
          name: basename.replace(/\.json$/i, ''),
          date: file.date ? new Date(file.date).toLocaleString() : 'Unknown date',
        };
      });
    });
  }

  /**
   * Handles creating a new save file.
   * Validates the filename and saves the scene.
   */
  _handleNewSave() {
    const input    = this._menuElement.querySelector('#save-new-filename');
    let   filename = input.value.trim();

    if (!filename) {
      CosmoScout.notifications.print('Could not save!', 'Please enter a file name.', 'warning');
      return;
    }

    if (!filename.endsWith('.json')) {
      filename += '.json';
    }

    CosmoScout.callbacks.core.save(filename)
        .then(() => {
          CosmoScout.notifications.print('Saved', `Scene saved as '${filename}'`, 'archive');
          input.value = '';
          this._renderSaveMenu();
        });
  }

  /**
   * Handles overwriting an existing save file.
   * @param {number} index - The index of the save file to overwrite.
   */
  _handleOverwriteSave(index) {
    CosmoScout.callbacks.core.getSaveFiles().then((saveFiles) => {
      if (index >= 0 && index < saveFiles.length) {
        const name     = saveFiles[index].name;
        const basename = name.split(/[\\/]/).pop();
        CosmoScout.callbacks.core.save(basename)
            .then(() => {
              CosmoScout.notifications.print(
                  'Saved', `Scene overwritten: '${basename}'`, 'archive');
              this._renderSaveMenu();
            });
      }
    });
  }

  /**
   * Handles loading a scene from a save file.
   * @param {number} index - The index of the save file to load.
   */
  _handleLoadScene(index) {
    CosmoScout.callbacks.core.getSaveFiles().then((saveFiles) => {
      if (index >= 0 && index < saveFiles.length) {
        const name     = saveFiles[index].name;
        const basename = name.split(/[\\/]/).pop();
        CosmoScout.callbacks.core.load(basename)
            .then(() => {
              CosmoScout.notifications.print(
                  'Loaded', `Scene loaded: '${basename}'`, 'open_in_browser');
              this.back();
            });
      }
    });
  }

  /**
   * Renders the load scene menu.
   * Displays a list of available save files for loading.
   */
  _renderLoadMenu() {
    this._menuBodyElement.innerHTML = '';

    const loadMenu = CosmoScout.gui.loadTemplateContent('main-menu-load-template');
    if (loadMenu === false) {
      return;
    }

    this._menuBodyElement.appendChild(loadMenu);

    const existingSavesSection = CosmoScout.gui.loadTemplateContent('main-menu-load-existing-template');
    if (existingSavesSection === false) {
      return;
    }
    loadMenu.appendChild(existingSavesSection);

    this._loadSaveFiles().then((saveFiles) => {
      const loadList = existingSavesSection.querySelector('.load-list');

      if (saveFiles.length === 0) {
        const emptyTemplate = CosmoScout.gui.loadTemplateContent('main-menu-load-empty-template');
        if (emptyTemplate !== false) {
          loadList.appendChild(emptyTemplate);
        }
        return;
      }

      saveFiles.forEach((saveFile, index) => {
        const loadItem = CosmoScout.gui.loadTemplateContent('main-menu-load-item-template');
        if (loadItem === false) {
          return;
        }

        loadItem.dataset.loadIndex                            = index;
        loadItem.querySelector('.load-item-name').textContent = saveFile.name;
        loadItem.querySelector('.load-item-date').textContent = saveFile.date;

        loadList.appendChild(loadItem);
      });

      document.querySelectorAll('[data-load-action="load"]').forEach((button) => {
        button.addEventListener('click', (event) => {
          const loadIndex = event.currentTarget.dataset.loadIndex;
          this._handleLoadScene(loadIndex);
        });
      });
    });
  }

  /**
   * Renders the current menu.
   * Updates the title, icon, back button visibility, and menu body content.
   */
  render() {
    const menu = this._menus[this._currentMenu];

    this._menuElement.querySelector('#main-menu-title').textContent = menu.title;
    this._menuElement.querySelector('#main-menu-icon').textContent  = menu.icon;
    this._backButtonElement.hidden                                  = this._history.length === 0;

    this._menuBodyElement.innerHTML = '';

    if (menu.render) {
      menu.render();
      return;
    }

    this._renderMenuItems(menu, this._menuBodyElement);
  }

  /**
   * Closes the main menu.
   */
  close() {
    this._menuElement.close();
  }
}
