/**
 * Simplistic api interface containing a name field and init method
 */
class IApi {
    /**
     * Api Name
     *
     * @type {string}
     */
    name;

    /**
     * Called in CosmoScout.init
     */
    init() {
    };
}

/**
 * Api Container holding all registered apis.
 */
class CosmoScout {
    /**
     * @type {Map<string, Object>}
     * @private
     */
    static _apis = new Map();

    /**
     * Init a list of apis
     *
     * @param apis {IApi[]}
     */
    static init(apis) {
        apis.forEach(Api => {
            let instance = new Api();
            this.register(instance.name, instance);
            instance.init();
        });
    }

    static initInputs() {
        this.initDropDowns();
        this.initChecklabelInputs();
        this.initRadiolabelInputs();
        this.initTooltips();
    }

    static initDropDowns() {
        document.querySelectorAll('.simple-value-dropdown').forEach(dropdown => {
            if (typeof dropdown.selectpicker !== "undefined") {
                dropdown.selectpicker();
            }

            dropdown.addEventListener('change', event => {
                if (event.target !== null && event.target.id !== '') {
                    CosmoScout.callNative(event.target.id, event.target.value);
                }
            });
        });
    }

    static initChecklabelInputs() {
        document.querySelectorAll('.checklabel input').forEach(input => {
            input.addEventListener('change', event => {
                if (event.target !== null) {
                    CosmoScout.callNative(event.target.id, event.target.checked);
                }
            })
        });
    }

    static initRadiolabelInputs() {
        document.querySelectorAll('.radiolabel input').forEach(input => {
            input.addEventListener('change', event => {
                if (event.target !== null) {
                    CosmoScout.callNative(event.target.id);
                }
            })
        });
    }

    static initTooltips() {
        const config = {delay: 500, placement: 'auto', html: false};

        document.querySelectorAll('[data-toggle="tooltip"]').forEach(tooltip => {
            if (typeof tooltip.tooltip !== "undefined") {
                tooltip.tooltip(config)
            }
        });

        document.querySelectorAll('[data-toggle="tooltip-bottom"]').forEach(tooltip => {
            if (typeof tooltip.tooltip !== "undefined") {
                config.placement = 'bottom';
                tooltip.tooltip(config)
            }
        });
    }

    /**
     *Appends a script element to the body
     *
     * @param url {string} Absolute or local file path
     * @param init {Function} Method gets run on script load
     */
    static registerJavaScript(url, init) {
        const script = document.createElement('script');
        script.setAttribute('type', 'text/javascript');
        script.setAttribute('src', url);

        if (typeof init !== "undefined") {
            script.addEventListener('load', init);
            script.addEventListener('readystatechange', init);
        }

        document.body.appendChild(script);
    }

    /**
     * Removes a script element by url
     *
     * @param url {string}
     */
    static unregisterJavaScript(url) {
        document.querySelectorAll('script').forEach(element => {
            if (typeof element.src !== "undefined" &&
                (element.src === url || element.src === this._localizeUrl(url))) {
                document.body.removeChild(element);
            }
        });
    }

    /**
     * Appends a link stylesheet to the head
     *
     * @param url {string}
     */
    static registerCss(url) {
        const link = document.createElement('link');
        link.setAttribute('type', 'text/css');
        link.setAttribute('rel', 'stylesheet');
        link.setAttribute('href', url);

        document.head.appendChild(link);
    }

    /**
     * Removes a stylesheet by url
     *
     * @param url {string}
     */
    static unregisterCss(url) {
        document.querySelectorAll('link').forEach(element => {
            if (typeof element.href !== "undefined" &&
                (element.href === url || element.href === this._localizeUrl(url))) {
                document.head.removeChild(element);
            }
        });
    }

    /**
     * Initialize a noUiSlider
     *
     * @param id {string}
     * @param min {number}
     * @param max {number}
     * @param step {number}
     * @param start {number[]}
     */
    static initSlider(id, min, max, step, start) {
        const slider = document.getElementById(id);

        if (typeof noUiSlider === "undefined") {
            console.error(`'noUiSlider' is not defined.`);
            return;
        }

        noUiSlider.create(slider, {
            start: start,
            connect: (start.length === 1 ? "lower" : true),
            step: step,
            range: {'min': min, 'max': max},
            format: {
                to: function (value) {
                    return Format.beautifyNumber(value);
                },
                from: function (value) {
                    return Number(parseFloat(value));
                }
            }
        });

        slider.noUiSlider.on('slide', function (values, handle, unencoded) {
            if (Array.isArray(unencoded)) {
                CosmoScout.callNative(id, unencoded[handle], handle);
            } else {
                CosmoScout.callNative(id, unencoded, 0);
            }
        });
    }

    /**
     * Global entry point to call any method on any registered api
     *
     * @param api {string} Name of api
     * @param method {string} Method name
     * @param args {string|number|boolean|Function|Object} One or more arguments to pass to the method
     * @return {*}
     */
    static call(api, method, ...args) {
        if (this._apis.has(api)) {
            if (typeof (this._apis.get(api))[method] !== "undefined") {
                return (this._apis.get(api))[method](...args);
            } else {
                console.error(`'${method}' does not exist on api '${api}'.`);
            }
        } else {
            console.error(`Api '${api}' is not registered.`);
        }
    }

    /**
     * window.call_native wrapper
     *
     * @param fn {string}
     * @param args {any}
     * @return {*}
     */
    static callNative(fn, ...args) {
        return window.call_native(fn, ...args);
    }

    /**
     * Register an api object
     *
     * @param name {string}
     * @param api {Object}
     */
    static register(name, api) {
        window[name + 'Api'] = api;
        this._apis.set(name, api);
    }

    /**
     * Remove a registered api by name
     *
     * @param name {string}
     */
    static remove(name) {
        delete window[name + 'Api'];
        this._apis.delete(name);
    }

    /**
     * Get a registered api object
     *
     * @param name {string}
     * @return {Object}
     */
    static getApi(name) {
        return this._apis.get(name);
    }

    /**
     * Localizes a filename
     *
     * @param url {string}
     * @return {string}
     * @private
     */
    static _localizeUrl(url) {
        return `file://../share/resources/gui/${url}`
    }
}

class SidebarApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'sidebar';

    /**
     * @type {DocumentFragment}
     * @private
     */
    _tabTemplate;

    /**
     * @type {DocumentFragment}
     * @private
     */
    _sectionTemplate;

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
     *
     * @type {RegExp}
     * @private
     */
    _idRegex = new RegExp('\%ID\%', 'g');

    /**
     * Loads all templates and needed container refs
     */
    init() {
        this._tabTemplate = document.getElementById('sidebar-plugin-tab-template').content.cloneNode(true);
        this._sectionTemplate = document.getElementById('sidebar-settings-section-template').content.cloneNode(true);
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
        let tab = this._tabTemplate.cloneNode(true).firstElementChild;

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
        let tab = this._sectionTemplate.cloneNode(true).firstElementChild;

        let html = this._replaceMarkers(tab.innerHTML, this._makeId(sectionName), icon, content);

        tab.innerHTML = html
            .replace('%SECTION%', sectionName)
            .trim();

        this._settings.appendChild(tab);
    }

    addCelestialBody() {

    }

    /**
     * @see {_clearHtml}
     * @param id {string}
     */
    clearContainer(id) {
        this._clearHtml(id);
    }

    /**
     * TODO UNUSED
     * @see {_clearHtml}
     * @param id {string}
     */
    clearDropdown(id) {
        const element = this._clearHtml(id);

        if (element !== null && typeof element.selectpicker !== "undefined") {
            element.selectpicker('render');
        }
    }

    addLocation() {

    }

    setElevationDataCopyright(copyright) {

    }

    setMapDataCopyright(copyright) {

    }

    /**
     * Adds an option to a dropdown
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

            if (typeof dropdown.selectpicker !== "undefined") {
                dropdown.selectpicker('refresh');
            }
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
        const dropdown = document.getElementById(id);

        if (dropdown !== null && typeof dropdown.selectpicker !== "undefined") {
            dropdown.selectpicker('val', value);
        } else {
            console.warn(`Dropdown '${id}' not found, or 'SelectPicker' not active.`);
        }
    }

    /**
     * Set a noUiSlider value
     *
     * @param id {string} Slider ID
     * @param value {number} Value
     */
    setSliderValue(id, ...value) {
        const slider = document.getElementById(id);

        if (slider !== null && typeof slider.noUiSlider !== "undefined") {
            if (value.length === 1) {
                slider.noUiSlider.set(value[0]);
            } else {
                slider.noUiSlider.set(value);
            }
        } else {
            console.warn(`Slider '${id} 'not found or 'noUiSlider' not active.`);
        }
    }

    /**
     * TODO
     * @param name {string}
     * @param icon {string}
     */
    addMeasurementTool(name, icon) {
        const area = $('#measurement-tools');
        area.append(`<div class='col-4 center' style='padding: 0 5px'>
                    <label style="width: 100%; height: 100%">
                        <input id='set_tool_${icon}' type='checkbox' name='measurement-tool' class='radio-button' />
                        <div class='block btn glass'>
                            <i style="font-size: 60px" class='material-icons'>${icon}</i> 
                            <br> 
                            <span>${name}</span>
                        </div>
                    </label>
                </div>`);

        $('#set_tool_' + icon).change(function () {
            $('#measurement-tools .radio-button').not('#set_tool_' + icon).prop('checked', false);

            if (this.checked) {
                window.call_native('set_measurement_tool', name);
            } else {
                window.call_native('set_measurement_tool', 'none');
            }
        });

        /*        const area = document.getElementById('measurement-tools');
                let tool = document.getElementById('measurement-tools-template').content.cloneNode(true).firstElementChild;

                tool.innerHTML = this._replaceMarkers(tool.innerHTML, '', icon, name);
                area.appendChild(tool);

                tool.addEventListener('change', function () {
                    document.querySelectorAll('#measurement-tools .radio-button').forEach(node => {
                        node.checked = false;
                    });

                    $('#measurement-tools .radio-button').not('#set_tool_' + icon).prop('checked', false);

                    if (this.checked) {
                        window.call_native('set_measurement_tool', name);
                    } else {
                        window.call_native('set_measurement_tool', 'none');
                    }
                });*/

    }

    deselectMeasurementTool() {
        document.querySelectorAll('#measurement-tools .radio-button').forEach(node => {
            node.checked = false;
        });
    }

    addSharad() {

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
            .replace(this._idRegex, id)
            .replace('%CONTENT%', content)
            .replace('%ICON%', icon)
            .trim();
    }

    /**
     * Clear the innerHtml of an element if it exists
     *
     * @param id
     * @return {HTMLElement | null}
     * @private
     */
    _clearHtml(id) {
        const element = document.getElementById(id);

        if (element !== null) {
            element.innerHTML = '';
        } else {
            console.warn(`No element #${id} found.`);
        }

        return element;
    }


}

class TimelineApi extends IApi {

}

class NotificationApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'notifications';

    /**
     * @type {HTMLElement}
     * @private
     */
    _container;

    /**
     * @type {DocumentFragment}
     * @private
     */
    _template;

    /**
     * Set the container in which to place the notifications
     * @param container {string}
     */
    init(container = 'notifications') {
        this._container = document.getElementById(container);
        this._template = document.getElementById('notification-template').content.cloneNode(true);
    }

    /**
     * Adds a notification into the initialized notification container
     *
     * @param title {string} Title
     * @param content {string} Content
     * @param icon {string} Materialize Icon Name
     * @param flyTo {string} Optional flyto name which gets passed to 'fly_to'. Activated on click
     */
    printNotification(title, content, icon, flyTo) {
        if (this._container.children.length > 4) {
            let no = this._container.lastElementChild;

            clearTimeout(no.timer);

            this._container.removeChild(no);
        }

        let notification = this._makeNotification(title, content, icon);

        if (flyTo) {
            notification.classList.add('clickable');
            notification.addEventListener('click', () => {
                CosmoScout.call('flyto', 'flyTo', flyTo);
            });
        }

        notification.classList.add('show');
        notification.timer = setTimeout(() => {
            notification.classList.add('fadeout');
            this._container.removeChild(notification);
        }, 8000);

        this._container.prepend(notification);
    }

    /**
     * Creates the actual HTML Notification
     *
     * @param title {string}
     * @param content {string}
     * @param icon {string}
     * @return {HTMLDivElement}
     * @private
     */
    _makeNotification(title, content, icon = '') {
        let notification = this._template.cloneNode(true).firstElementChild;

        notification.innerHTML = notification.innerHTML
            .replace('%TITLE%', title)
            .replace('%CONTENT%', content)
            .replace('%ICON%', icon)
            .trim();

        return notification;
    }
}

class StatusbarApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'statusbar';

    /**
     * Element ids
     * @type {{userPosition: string, pointerPosition: string, speed: string}}
     */
    config = {
        userPosition: 'placeholder-5',
        pointerPosition: 'placeholder-6',
        speed: 'placeholder-8',
    };

    /**
     * @type {HTMLElement}
     * @private
     */
    _userContainer;

    /**
     * @type {HTMLElement}
     * @private
     */
    _pointerContainer;

    /**
     * @type {HTMLElement}
     * @private
     */
    _speedContainer;

    /**
     * Provide config object to overwrite default html container ids
     * @param config {{userPosition: string, pointerPosition: string, speed: string}}
     */
    constructor(config = {}) {
        super();

        Object.assign(this.config, config);
    }

    /**
     * Initialize all containers
     */
    init() {
        this._userContainer = document.getElementById(this.config.userPosition);
        this._pointerContainer = document.getElementById(this.config.pointerPosition);
        this._speedContainer = document.getElementById(this.config.speed);
    }

    /**
     * Set the current user position string
     *
     * @param long {number|string} Longitude
     * @param lat {number|string} Latitude
     * @param height {number|string} Height
     */
    setUserPosition(long, lat, height) {
        this._userContainer.innerText = Format.longitude(long) + Format.latitude(lat) + "(" + Format.height(height) + ")";
    }

    /**
     * Set the current pointer position
     *
     * @param hits {boolean} True if the pointer ray hits an object
     * @param long {number|string} Longitude
     * @param lat {number|string} Latitude
     * @param height {number|string} Height
     */
    setPointerPosition(hits, long, lat, height) {
        let text = ' - ';

        if (hits) {
            text = Format.longitude(long) + Format.latitude(lat) + "(" + Format.height(height) + ")";
        }

        this._pointerContainer.innerText = text;
    }

    /**
     * Set the current navigator speed
     *
     * @param speed {number}
     */
    setSpeed(speed) {
        this._speedContainer.innerText = Format.speed(speed);
    }
}

class StatisticsApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'statistics';

    _values = [];
    _colorHash;
    _minTime = 1000;
    _maxValue = 1e9 / 30;
    _alpha = 0.95;

    constructor() {
        super();

        if (typeof ColorHash !== "undefined") {
            this._colorHash = new ColorHash({lightness: 0.5, saturation: 0.3})
        } else {
            console.error(`Class 'ColorHash' not defined.`);
        }
    }

    _resetTimes(data) {
        this._values.forEach(value => {
            if (typeof data[value.name] !== "undefined") {
                value.timeGPU = data[value.name][0];
                value.timeCPU = data[value.name][1];
                data[value.name][0] = -1;
                data[value.name][1] = -1;
            } else {
                value.timeGPU = 0;
                value.timeCPU = 0;
            }
        });

        return data;
    }

    _addNewElements(data) {
        for (let key in data) {
            if (!data.hasOwnProperty(key)) {
                continue;
            }

            if (data[key][0] >= 0) {
                this._values.push({
                    "name": key,
                    "timeGPU": data[key][0],
                    "timeCPU": data[key][1],
                    "avgTimeGPU": data[key][0],
                    "avgTimeCPU": data[key][1],
                    "color": this._colorHash.hex(key)
                });
            }
        }
    }

    /**
     *
     * @param data {string}
     * @param frameRate {number}
     */
    setData(data, frameRate) {
        this._values = [];
        data = JSON.parse(data);

        // first set all times to zero
        data = this._resetTimes(data);


        // then add all new elements
        this._addNewElements(data);


        // remove all with very little contribution
        this._values = this._values.filter(element => {
            return element.timeGPU > this._minTime || element.timeCPU > this._minTime ||
                element.avgTimeGPU > this._minTime || element.avgTimeCPU > this._minTime;
        });

        // update average values
        this._values.forEach(element => {
            element.avgTimeGPU = element.avgTimeGPU * this._alpha + element.timeGPU * (1 - this._alpha);
            element.avgTimeCPU = element.avgTimeCPU * this._alpha + element.timeCPU * (1 - this._alpha);
        });


        // sort by average
        this._values.sort((a, b) => {
            return (b.avgTimeGPU + b.avgTimeCPU) - (a.avgTimeGPU + a.avgTimeCPU);
        });

        this._insertHtml(frameRate);
    }

    _insertHtml(frameRate) {
        const container = document.getElementById('statistics');
        container.innerHTML = '';

        const maxEntries = Math.min(10, this._values.length);
        const maxWidth = container.offsetWidth;

        container.innerHTML += `<div class="label"><strong>FPS: ${frameRate.toFixed(2)}</strong></div>`;
        /*        for (let i = 0; i < maxEntries; ++i) {
                    const widthGPU = maxWidth * this._values[i].avgTimeGPU / this._maxValue;
                    const widthCPU = maxWidth * this._values[i].avgTimeCPU / this._maxValue;

                    container.innerHTML += `<div class="item">
                    <div class="bar gpu" style="background-color:${this._values[i].color}; width:${widthGPU}px"><div class='label'>gpu: ${(this._values[i].avgTimeGPU * 0.000001).toFixed(1)} ms</div></div>
                    <div class="bar cpu" style="background-color:${this._values[i].color}; width:${widthCPU}px"><div class='label'>cpu: ${(this._values[i].avgTimeCPU * 0.000001).toFixed(1)} ms</div></div>
                    <div class='label'>${this._values[i].name}</div>
                </div>`;

                }*/
    }

}

class FlyToApi extends IApi {
    name = 'flyto';

    flyTo(planet, location, time) {
        let name;

        if (typeof location === "undefined") {
            CosmoScout.callNative('fly_to', planet);
        } else {
            CosmoScout.callNative('fly_to', planet + '', location.longitude + '', location.latitude + '', location.height + '', time + '');
        }

        CosmoScout.call('notifications', 'printNotification', 'Traveling', `to ${planet}`, 'send');
    }

    setCelestialBody(name) {
        CosmoScout.callNative('set_celestial_body', name);
    }
}

/**
 * Formats different numbers
 */
class Format {
    /**
     * @param number {number|string}
     * @return {string}
     */
    static number(number) {
        number = parseFloat(number);

        // Set very small numbers to 0
        if (number < Number.EPSILON && -Number.EPSILON > number) {
            number = 0;
        }

        if (Math.abs(number) < 10) {
            return number.toFixed(2)
        } else if (Math.abs(number) < 100) {
            return number.toFixed(1)
        }

        return number.toFixed(0)
    }

    /**
     * Returns a formatted height string
     *
     * @param height {number|string}
     * @return {string}
     */
    static height(height) {
        let num;
        let unit;

        height = parseFloat(height);

        if (Math.abs(height) < 0.1) {
            num = Format.number(height * 1000);
            unit = 'mm';
        } else if (Math.abs(height) < 1) {
            num = Format.number(height * 100);
            unit = 'cm';
        } else if (Math.abs(height) < 1e4) {
            num = Format.number(height);
            unit = 'm';
        } else if (Math.abs(height) < 1e7) {
            num = Format.number(height / 1e3);
            unit = 'km';
        } else if (Math.abs(height) < 1e10) {
            num = Format.number(height / 1e6);
            unit = 'Tsd km';
        } else if (Math.abs(height / 1.496e11) < 1e4) {
            num = Format.number(height / 1.496e11);
            unit = 'AU';
        } else if (Math.abs(height / 9.461e15) < 1e3) {
            num = Format.number(height / 9.461e15);
            unit = 'ly';
        } else if (Math.abs(height / 3.086e16) < 1e3) {
            num = Format.number(height / 3.086e16);
            unit = 'pc';
        } else {
            num = Format.number(height / 3.086e19);
            unit = 'kpc';
        }

        return `${num} ${unit}`;
    }

    /**
     * Returns a formatted speed string
     *
     * @param speed {number|string}
     * @return {string}
     */
    static speed(speed) {
        let num;
        let unit;

        speed = parseFloat(speed);

        if (Math.abs(speed * 3.6) < 500) {
            num = Format.number(speed * 3.6);
            unit = 'km/h';
        } else if (Math.abs(speed) < 1e3) {
            num = Format.number(speed);
            unit = 'm/s';
        } else if (Math.abs(speed) < 1e7) {
            num = Format.number(speed / 1e3);
            unit = 'km/s';
        } else if (Math.abs(speed) < 1e8) {
            num = Format.number(speed / 1e6);
            unit = 'Tsd km/s';
        } else if (Math.abs(speed / 2.998e8) < 1e3) {
            num = Format.number(speed / 2.998e8);
            unit = 'SoL';
        } else if (Math.abs(speed / 1.496e11) < 1e3) {
            num = Format.number(speed / 1.496e11);
            unit = 'AU/s';
        } else if (Math.abs(speed / 9.461e15) < 1e3) {
            num = Format.number(speed / 9.461e15);
            unit = 'ly/s';
        } else if (Math.abs(speed / 3.086e16) < 1e3) {
            num = Format.number(speed / 3.086e16);
            unit = 'pc/s';
        } else {
            num = Format.number(speed / 3.086e19);
            unit = 'kpc/s';
        }

        return `${num} ${unit}`;
    }

    /**
     * Returns a formatted latitude string
     *
     * @param lat {number|string}
     * @return {string}
     */
    static latitude(lat) {
        lat = parseFloat(lat);

        if (lat < 0) {
            return (-lat).toFixed(2) + "° S ";
        }

        return (lat).toFixed(2) + "° N "
    }

    /**
     * Returns a formatted longitude string
     *
     * @param lon {number|string}
     * @return {string}
     */
    static longitude(lon) {
        lon = parseFloat(lon);

        if (lon < 0) {
            return (-lon).toFixed(2) + "° W ";
        }

        return (lon).toFixed(2) + "° E ";
    }

    /**
     *
     * @param number {number}
     * @return {string|number}
     */
    static beautifyNumber(number) {
        const abs = Math.abs(number);
        let value;

        if (abs >= 10000) {
            value = Number(number.toPrecision(2)).toExponential()
        } else if (abs >= 1000) {
            value = Number(number.toPrecision(4))
        } else if (abs >= 1) {
            value = Number(number.toPrecision(3))
        } else if (abs >= 0.1) {
            value = Number(number.toPrecision(2))
        } else if (abs === 0) {
            value = '0';
        } else {
            value = Number(number.toPrecision(2)).toExponential();
        }

        return value.toString();
    }
}

/**
 * Locales won't work in Android WebView
 */
class DateOperations {
    static _defaultLocale;

    /**
     * Set a locale for all formatDateReadable calls
     *
     * @param locale
     */
    static setLocale(locale) {
        try {
            new Date().toLocaleString('i');
        } catch (e) {
            console.error('Browser does not support setting date locales.');

            return;
        }

        this._defaultLocale = locale;
    }

    /**
     *Format a Date to a for a human readable string DD.MM.YYYY HH:MM:SS
     *
     * @param date {Date}
     * @return {string}
     */
    static formatDateReadable(date) {
        return `${date.toLocaleDateString(this._defaultLocale)} ${date.toLocaleTimeString(this._defaultLocale)}`;
    }

    /**
     * Format a Date to YYYY-MM-DD
     *
     * @param date {Date}
     * @return {string}
     */
    static getFormattedDate(date) {
        return date.toISOString().split('T')[0];
    }

    /**
     * Format a Date to YYYY-MM-DD HH:MM:SS
     *
     * @param date {Date}
     * @return {string}
     */
    static getFormattedDateWithTime(date) {
        return `${this.getFormattedDate(date)} ${date.toLocaleTimeString('de-de')}`;
    }

    /**
     * Format a Date to a readable format for CosmoScoutVR YYYY-MM-DD HH:MM:SS.sss
     *
     * @param date {Date}
     * @return {string}
     */
    static formatDateCosmo(date) {
        let milli = date.getMilliseconds().toString().padStart(3, '0');

        return `${this.getFormattedDateWithTime(date)}.${milli}`;
    }

    /**
     * Convert seconds into an object containing the duration in hours -- ms
     *
     * @param seconds {number}
     * @return {{}}
     */
    static convertSeconds(seconds) {
        const mSec = 60;
        const hSec = mSec * mSec;
        const dSec = hSec * 24;

        let converted = {};

        converted.days = Math.floor(seconds / dSec);
        converted.hours = Math.floor((seconds - (converted.days * dSec)) / hSec);
        converted.minutes = Math.floor((seconds - (converted.days * dSec) - (converted.hours * hSec)) / mSec);
        converted.seconds = Math.floor(seconds - (converted.days * dSec) - (converted.hours * hSec) - (converted.minutes * mSec));
        converted.milliSec = Math.round((seconds - Math.floor(seconds)) * 1000);

        return converted;
    }

    /**
     * Increase a Date by days, hours , minutes, seconds and milliseconds
     *
     * @param date {Date}
     * @param days {number}
     * @param hours {number}
     * @param minutes {number}
     * @param seconds {number}
     * @param milliSec {number}
     * @return {Date}
     */
    static increaseDate(date, days, hours, minutes, seconds, milliSec) {
        date.setDate(date.getDate() + days);
        date.setHours(date.getHours() + hours);
        date.setMinutes(date.getMinutes() + minutes);
        date.setSeconds(date.getSeconds() + seconds);
        date.setMilliseconds(date.getMilliseconds() + milliSec);
        return date;
    }

    /**
     * Decrease a Date by days, hours , minutes, seconds and milliseconds
     *
     * @param date {Date}
     * @param days {number}
     * @param hours {number}
     * @param minutes {number}
     * @param seconds {number}
     * @param milliSec {number}
     * @return {Date}
     */
    static decreaseDate(date, days, hours, minutes, seconds, milliSec) {
        return this.increaseDate(date, -days, -hours, -minutes, -seconds, -milliSec);
    }
}