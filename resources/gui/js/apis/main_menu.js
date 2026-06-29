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
                        action: () => {
                            // TODO: save callback
                        },
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
                render: () => {
                    CosmoScout.callbacks.core.getPlugins().then((plugins) => {
                        const pluginManager = document.querySelector('.plugin-manager');
                        pluginManager.innerHTML = '';
                        Object.entries(plugins).forEach(([name, active]) => {
                            const pluginItem = document.createElement('div');
                            pluginItem.classList.add('plugin-manager-item', 'row');
                            if (active)
                                pluginItem.innerHTML = `
                                    <span class="plugin-manager-item-name col-10">${name}</span>
                                    <a class="btn light-glass plugin-manager-item-action col-1" data-index="${name}"
                                       onclick="CosmoScout.callbacks.core.reloadPlugin('${name}').then(() => CosmoScout.mainMenu.render())">
                                        <i class="material-icons">refresh</i>
                                    </a>
                                    <a class="btn light-glass plugin-manager-item-action col-1" data-index="${name}"
                                       onclick="CosmoScout.callbacks.core.unloadPlugin('${name}').then(() => CosmoScout.mainMenu.render())">
                                        <i class="material-icons">extension_off</i>
                                    </a>
                                `;
                            else
                                pluginItem.innerHTML = `
                                    <span class="plugin-manager-item-name col-10">${name}</span>
                                    <a class="btn light-glass plugin-manager-item-action col-2" data-index="${name}"
                                       onclick="CosmoScout.callbacks.core.loadPlugin('${name}').then(() => CosmoScout.mainMenu.render())">
                                        <i class="material-icons">extension</i>
                                    </a>
                                `;
                            pluginManager.appendChild(pluginItem);
                        });
                    })
                    return `
                      <div class="plugin-manager container"></div>
                    `
                },
            },
        };

        document
            .querySelector('#main-menu-back-button')
            .addEventListener('click', () => this.back());

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

    render() {
        const menu = this._menus[this._currentMenu];

        document.querySelector('#main-menu-title').textContent = menu.title;
        document.querySelector('#main-menu-icon').textContent = menu.icon;
        document.querySelector('#main-menu-back-button').hidden = this._history.length === 0;

        const body = document.querySelector('#main-menu-body');

        if (menu.render) {
            body.innerHTML = menu.render();
            return;
        }

        body.innerHTML = menu.items
            .map((item, index) => `
                <a class="btn light-glass col-12 main-menu-item" data-index="${index}">
                  ${item.icon ? `<i class="material-icons">${item.icon}</i>` : ''}
                  <span>${item.label}</span>
                </a>
            `)
            .join('');

        body.querySelectorAll('.main-menu-item').forEach((element) => {
            element.addEventListener('click', () => {
                const item = menu.items[Number(element.dataset.index)];

                if (item.target) {
                    this.navigate(item.target);
                } else if (item.action) {
                    item.action();
                }
            });
        });
    }
}
