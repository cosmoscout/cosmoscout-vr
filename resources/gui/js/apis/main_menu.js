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
                        action: () => {
                            // TODO: load callback
                        },
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
        };

        document
            .querySelector('#main-menu-back-button')
            .addEventListener('click', () => this.back());

        document
            .querySelector('#main-menu-body')
            .addEventListener('click', (event) => this._handleBodyClick(event));

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
        const body = document.querySelector('#main-menu-body');
        const token = ++this._renderToken;

        body.innerHTML = '';

        const pluginManager = this._loadTemplate('main-menu-plugin-template');
        if (pluginManager === false) {
            return '';
        }

        body.appendChild(pluginManager);

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
        const body = document.querySelector('#main-menu-body');
        body.innerHTML = '';

        const saveMenu = this._loadTemplate('main-menu-save-template');
        if (saveMenu === false) {
            return '';
        }

        body.appendChild(saveMenu);

        return '';
    }

    render() {
        const menu = this._menus[this._currentMenu];

        document.querySelector('#main-menu-title').textContent = menu.title;
        document.querySelector('#main-menu-icon').textContent = menu.icon;
        document.querySelector('#main-menu-back-button').hidden = this._history.length === 0;

        const body = document.querySelector('#main-menu-body');
        body.innerHTML = '';
        this._renderToken += 1;

        if (menu.render) {
            menu.render();
            return;
        }

        this._renderMenuItems(menu, body);
    }
}
