////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

/**
 * The main menu
 */
class MainMenuApi extends IApi {
    name = 'mainMenu';

    init() {
        this._currentMenu = 'root';
        this._history = [];
        this._renderToken = 0;

        this._menuElement = document.querySelector('#cs-main-menu');
        this._menuBodyElement = this._menuElement.querySelector('#main-menu-body');
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

        this._menuBodyElement
            .addEventListener('click', (event) => this._handleBodyClick(event));

        this._menuElement
            .addEventListener('click', (event) => {
                if (event.target === event.currentTarget) {
                    this.close();
                }
            });

        this.render();
    }

    navigate(menuId) {
        if (!this._menus[menuId]) {
            console.warn(`Unknown menu: ${menuId}`);
            return;
        }

        this._history.push(this._currentMenu);
        this._currentMenu = menuId;
        this.render();
    }

    back() {
        if (this._history.length === 0) {
            return;
        }

        this._currentMenu = this._history.pop();
        this.render();
    }

    _loadTemplate(id) {
        const template = CosmoScout.gui.loadTemplateContent(id);

        if (template === false) {
            console.warn(`Menu template '${id}' could not be loaded.`);
        }

        return template;
    }

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
            const action = pluginAction.dataset.pluginAction;

            this._runPluginAction(pluginName, action);
        }
    }

    _runPluginAction(pluginName, action) {
        if (!pluginName || !action) {
            return;
        }

        let callback = null;

        if (action === 'reload') {
            callback = CosmoScout.callbacks.core.reloadPlugin(pluginName);
        } else if (action === 'unload') {
            callback = CosmoScout.callbacks.core.unloadPlugin(pluginName);
        } else if (action === 'load') {
            callback = CosmoScout.callbacks.core.loadPlugin(pluginName);
        }

        if (!callback) {
            console.warn(`Unknown plugin action: ${action}`);
            return;
        }

        callback.then(() => this.render());
    }

    _renderMenuItems(menu, body) {
        menu.items.forEach((item, index) => {
            const element = this._loadTemplate('main-menu-item-template');
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

    _renderPlugins() {
        const token = ++this._renderToken;

        this._menuBodyElement.innerHTML = '';

        const pluginManager = this._loadTemplate('main-menu-plugin-template');
        if (pluginManager === false) {
            return '';
        }

        this._menuBodyElement.appendChild(pluginManager);

        CosmoScout.callbacks.core.getPlugins().then((plugins) => {
            if (token !== this._renderToken || this._currentMenu !== 'plugins') {
                return;
            }

            const pluginList = pluginManager.querySelector('.plugin-manager-items');
            pluginList.innerHTML = '';

            Object.entries(plugins).forEach(([name, active]) => {
                const templateId = active
                    ? 'main-menu-plugin-item-active-template'
                    : 'main-menu-plugin-item-inactive-template';
                const pluginItem = this._loadTemplate(templateId);

                if (pluginItem === false) {
                    return;
                }

                pluginItem.dataset.pluginName = name;
                pluginItem.querySelector('.plugin-manager-item-name').textContent = name;

                pluginItem
                    .querySelectorAll('[data-plugin-action]')
                    .forEach((actionButton) => {
                        actionButton.dataset.pluginName = name;
                    });

                pluginList.appendChild(pluginItem);
            });
        });

        return '';
    }

    _renderSaveMenu() {
        this._menuBodyElement.innerHTML = '';

        const saveMenu = this._loadTemplate('main-menu-save-template');
        if (saveMenu === false) {
            return '';
        }

        this._menuBodyElement.appendChild(saveMenu);

        const newFileSection = document.createElement('div');
        newFileSection.className = 'save-new-file mb-3';
        newFileSection.innerHTML = `
            <div class="row mb-2">
                <div class="col-12">
                    <label class="form-label">Save New Scene</label>
                </div>
            </div>
            <div class="row">
                <div class="col-8">
                    <input type="text" class="form-control" id="save-new-filename" placeholder="Enter file name...">
                </div>
                <div class="col-4">
                    <button class="btn glass block" id="save-new-button">
                        <i class="material-icons">save</i> Save
                    </button>
                </div>
            </div>
        `;
        saveMenu.appendChild(newFileSection);

        const existingSavesSection = document.createElement('div');
        existingSavesSection.className = 'save-existing-files';
        existingSavesSection.innerHTML = `
            <div class="row mb-2">
                <div class="col-12">
                    <label class="form-label">Overwrite Existing Scene</label>
                </div>
            </div>
            <div class="row">
                <div class="col-12">
                    <div class="save-list"></div>
                </div>
            </div>
        `;
        saveMenu.appendChild(existingSavesSection);

        this._loadSaveFiles().then((saveFiles) => {
            const saveList = existingSavesSection.querySelector('.save-list');

            if (saveFiles.length === 0) {
                saveList.innerHTML = '<div class="text-muted text-center py-3">No saved scenes found</div>';
                return;
            }

            saveFiles.forEach((saveFile, index) => {
                const saveItem = this._loadTemplate('main-menu-save-item-template');
                if (saveItem === false) {
                    return;
                }

                saveItem.dataset.saveIndex = index;
                saveItem.querySelector('.save-item-name').textContent = saveFile.name;
                saveItem.querySelector('.save-item-date').textContent = saveFile.date;

                saveList.appendChild(saveItem);
            });

            document
                .querySelectorAll('[data-save-action="overwrite"]')
                .forEach((button) => {
                    button.addEventListener('click', (event) => {
                        const saveIndex = event.currentTarget.dataset.saveIndex;
                        this._handleOverwriteSave(saveIndex);
                    });
                });

            document
                .querySelector('#save-new-button')
                .addEventListener('click', () => this._handleNewSave());
        });

        return '';
    }

    _loadSaveFiles() {
        return CosmoScout.callbacks.core.getSaveFiles().then((saveFiles) => {
            return saveFiles.map((file) => {
                const name = file.name || 'Untitled';
                const basename = name.split(/[\\/]/).pop();
                return {
                    name: basename.replace(/\.json$/i, ''),
                    date: file.date ? new Date(file.date).toLocaleString() : 'Unknown date',
                };
            });
        });
    }

    _handleNewSave() {
        const input = this._menuElement.querySelector('#save-new-filename');
        let filename = input.value.trim();

        if (!filename) {
            CosmoScout.notifications.print('Could not save!', 'Please enter a file name.', 'warning');
            return;
        }

        if (!filename.endsWith('.json')) {
            filename += '.json';
        }

        CosmoScout.callbacks.core.save(filename).then(() => {
            CosmoScout.notifications.print('Saved', `Scene saved as "${filename}"`, 'archive');
            input.value = '';
            this._renderSaveMenu();
        }).catch((err) => {
            CosmoScout.notifications.print('Error', `Failed to save scene: ${err.message || err}`, 'error');
        });
    }

    _handleOverwriteSave(index) {
        CosmoScout.callbacks.core.getSaveFiles().then((saveFiles) => {
            if (index >= 0 && index < saveFiles.length) {
                const name = saveFiles[index].name;
                const basename = name.split(/[\\/]/).pop();
                CosmoScout.callbacks.core.save(basename).then(() => {
                    CosmoScout.notifications.print('Saved', `Scene overwritten: "${basename}"`, 'archive');
                    this._renderSaveMenu();
                }).catch((err) => {
                    CosmoScout.notifications.print('Error', `Failed to overwrite scene: ${err.message || err}`, 'error');
                });
            }
        });
    }

    _handleLoadScene(index) {
        CosmoScout.callbacks.core.getSaveFiles().then((saveFiles) => {
            if (index >= 0 && index < saveFiles.length) {
                const name = saveFiles[index].name;
                const basename = name.split(/[\\/]/).pop();
                CosmoScout.callbacks.core.load(basename).then(() => {
                    CosmoScout.notifications.print('Loaded', `Scene loaded: "${basename}"`, 'open_in_browser');
                    this.back();
                }).catch((err) => {
                    CosmoScout.notifications.print('Error', `Failed to load scene: ${err.message || err}`, 'error');
                });
            }
        });
    }

    _renderLoadMenu() {
        this._menuBodyElement.innerHTML = '';

        const loadMenu = this._loadTemplate('main-menu-load-template');
        if (loadMenu === false) {
            return '';
        }

        this._menuBodyElement.appendChild(loadMenu);

        const existingSavesSection = document.createElement('div');
        existingSavesSection.className = 'load-existing-files';
        existingSavesSection.innerHTML = `
            <div class="row mb-2">
                <div class="col-12">
                    <label class="form-label">Load Scene</label>
                </div>
            </div>
            <div class="row">
                <div class="col-12">
                    <div class="load-list"></div>
                </div>
            </div>
        `;
        loadMenu.appendChild(existingSavesSection);

        this._loadSaveFiles().then((saveFiles) => {
            const loadList = existingSavesSection.querySelector('.load-list');

            if (saveFiles.length === 0) {
                loadList.innerHTML = '<div class="text-muted text-center py-3">No saved scenes found</div>';
                return;
            }

            saveFiles.forEach((saveFile, index) => {
                const loadItem = this._loadTemplate('main-menu-load-item-template');
                if (loadItem === false) {
                    return;
                }

                loadItem.dataset.loadIndex = index;
                loadItem.querySelector('.load-item-name').textContent = saveFile.name;
                loadItem.querySelector('.load-item-date').textContent = saveFile.date;

                loadList.appendChild(loadItem);
            });

            document
                .querySelectorAll('[data-load-action="load"]')
                .forEach((button) => {
                    button.addEventListener('click', (event) => {
                        const loadIndex = event.currentTarget.dataset.loadIndex;
                        this._handleLoadScene(loadIndex);
                    });
                });
        });

        return '';
    }

    render() {
        const menu = this._menus[this._currentMenu];

        this._menuElement.querySelector('#main-menu-title').textContent = menu.title;
        this._menuElement.querySelector('#main-menu-icon').textContent = menu.icon;
        this._backButtonElement.hidden = this._history.length === 0;

        this._menuBodyElement.innerHTML = '';
        this._renderToken += 1;

        if (menu.render) {
            menu.render();
            return;
        }

        this._renderMenuItems(menu, this._menuBodyElement);
    }

    close() {
        this._menuElement.close();
    }
}
