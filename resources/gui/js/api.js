/**
 * Simplistic api interface containing a name field and init method
 */
import * as vis from "../third-party/js/vis-timeline-graph2d.min";

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

    /**
     * Initialize third party drop downs,
     * add input event listener,
     * initialize tooltips
     */
    static initInputs() {
        this.initDropDowns();
        this.initChecklabelInputs();
        this.initRadiolabelInputs();
        this.initTooltips();
        this.initDataCalls();
    }

    /**
     * @see {initInputs}
     */
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

    /**
     * @see {initInputs}
     */
    static initChecklabelInputs() {
        document.querySelectorAll('.checklabel input').forEach(input => {
            input.addEventListener('change', event => {
                if (event.target !== null) {
                    CosmoScout.callNative(event.target.id, event.target.checked);
                }
            })
        });
    }

    /**
     * @see {initInputs}
     */
    static initRadiolabelInputs() {
        document.querySelectorAll('.radiolabel input').forEach(input => {
            input.addEventListener('change', event => {
                if (event.target !== null) {
                    CosmoScout.callNative(event.target.id);
                }
            })
        });
    }

    /**
     * @see {initInputs}
     * Adds an onclick listener to every element containing [data-call="methodname"]
     * The method name gets passed to call_native
     */
    static initDataCalls() {
        document.querySelectorAll('[data-call]').forEach(input => {
            input.addEventListener('click', () => {
                console.log(input.dataset);
                if (typeof input.dataset.call !== "undefined") {
                    CosmoScout.callNative(input.dataset.call);
                }
            })
        });
    }

    /**
     * @see {initInputs}
     */
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
     * Appends a script element to the body
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
        if (method !== 'setUserPosition' && method !== 'setNorthDirection' && method !== 'setSpeed') {
            console.log(`Calling '${method}' on '${api}'`);
        }

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
        this[name] = api;
        this._apis.set(name, api);
    }

    /**
     * Remove a registered api by name
     *
     * @param name {string}
     */
    static remove(name) {
        delete this[name];
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
    /**print_notification
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

    addCelestialBody(name, icon) {
        const area = $('#celestial-bodies');
        area.append(`<div class='col-3 center' style='padding: 3px'>
                    <a class='block btn glass' id='set_body_${name}'>
                        <img style='pointer-events: none' src='../icons/${icon}' height='80' width='80'>
                        ${name}
                    </a>
                </div>`);

        $('#set_body_' + name).on('click', function () {
            window.call_native('set_celestial_body', name);
        });
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

    /**
     * TODO
     * @param group
     * @param text
     */
    addLocation(group, text) {
        let first = false;
        const tabArea = $("#location-tabs-area");
        if (tabArea.children().length === 0) {
            first = true;
            tabArea.append(`
            <nav>
                <div class="row nav nav-tabs" id="location-tabs" role="tablist"></div>
            </nav>
            <div class="tab-content" id="nav-tabContents"></div>
        `)
        }

        const locationsTab = $("#location-tabs");
        const tabContents = $("#nav-tabContents");

        let groupTab = $(`#nav-${group}`);
        if (groupTab.length === 0) {
            const active = first ? "active" : "";
            locationsTab.append(`
            <a class="nav-item nav-link ${active} col-4" id="nav-${group}-tab" data-toggle="tab" href="#nav-${group}" 
                role="tab" aria-controls="nav-${group}" aria-selected="${first}">${group}</a>
        `);

            const show = first ? "show" : "";

            tabContents.append(`
            <div class="tab-pane fade ${show} ${active}" id="nav-${group}" role="tabpanel" aria-labelledby="nav-${group}-tab"></div>
        `);

            groupTab = $(`#nav-${group}`);
        }

        groupTab.append(`
        <div class='row'>
            <div class='col-8'>
                ${text}
            </div>
            <div class='col-4'>
                <a class='btn glass block fly-to' data-toggle="tooltip" title='Fly to ${text}' onclick='window.call_native("fly_to", "${text}")'>
                    <i class='material-icons'>send</i>
                </a>
            </div>
        </div>
    `);

        $('[data-toggle="tooltip"]').tooltip({delay: 500, placement: "auto", html: false});
    }

    setElevationDataCopyright(copyright) {
        $("#img-data-copyright").tooltip({title: "© " + copyright, placement: "top"});

    }

    setMapDataCopyright(copyright) {
        $("#dem-data-copyright").tooltip({title: "© " + copyright, placement: "bottom"});
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

    /**
     * TODO
     * @param file
     * @param time
     */
    addSharad(file, time) {
        const html = `
        <div class='row item-${file}''>
            <div class='col-8' >${file}</div>
            <div class='col-4'><a class='btn glass block' onclick='window.call_native("set_time", ${time})' ><i class='material-icons'>restore</i></a></div>
        </div>`;

        $('#list-sharad').append(html);
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

class VisTimelineEvent {
    /**
     * @type {Number|null}
     */
    group;

    /**
     * @type {Number|null}
     */
    item;

    /**
     * @type {Number|null}
     */
    customTime;

    /**
     * @type {Number}
     */
    pageX;

    /**
     * @type {Number}
     */
    pageY;

    /**
     * @type {Number}
     */
    x;

    /**
     * @type {Number}
     */
    y;

    /**
     * @type {Date}
     */
    time;


    /**
     * @type {Date}
     */
    snappedTime;

    /**
     * @type {string|null}
     */
    what;

    /**
     * @type {Event}
     */
    event;


    /* Select Event */
    /**
     * @type {Number[]}
     */
    items;
}

class TimelineApi extends IApi {
    name = 'timeline';

    PAUSE = 0;
    REALTIME = 1;
    MINUTES = 60;
    HOURS = 3600;
    DAYS = 86400;
    MONTHS = 2628000;


    _buttonTemplate;
    _buttonContainer;

    /**
     * @type {DataSet}
     */
    _items;

    /**
     * @type {DataSet}
     */
    _itemsOverview;

    _pauseOptions = {
        moveable: true,
        zoomable: true,
    };

    _playingOptions = {
        moveable: false,
        zoomable: false,
    };

    /**
     * @type {Timeline}
     */
    _timeline;

    _timelineOptions = {
        minHeight: 35,
        maxHeight: 35,
        stack: false,
        max: new Date(2030, 12),
        min: new Date(1950, 1),
        zoomable: false,
        moveable: false,
        showCurrentTime: false,
        editable: {
            add: true,         // add new items by double tapping
            updateTime: true,  // drag items horizontally
            updateGroup: false, // drag items from one group to another
            remove: false,       // delete an item by tapping the delete button top right
            overrideItems: false  // allow these options to override item.editable
        },
        onAdd: on_add_callback,
        onUpdate: on_update_callback,
        onMove: on_item_move_callback,
        format: {
            minorLabels: {
                millisecond: 'SSS[ms]',
                second: 's[s]',
                minute: 'HH:mm',
                hour: 'HH:mm',
                weekday: 'ddd D',
                day: 'ddd D',
                week: 'MMM D',
                month: 'MMM',
                year: 'YYYY'
            },
            majorLabels: {
                millisecond: 'HH:mm:ss',
                second: 'D MMMM HH:mm',
                minute: 'ddd D MMMM',
                hour: 'ddd D MMMM',
                weekday: 'MMMM YYYY',
                day: 'MMMM YYYY',
                week: 'MMMM YYYY',
                month: 'YYYY',
                year: ''
            }
        }
    };

    /**
     * @type {Timeline}
     */
    _overviewTimeline;

    _overviewTimelineOptions = {
        minHeight: 40,
        maxHeight: 40,
        stack: false,
        max: new Date(2030, 12),
        min: new Date(1950, 1),
        zoomable: true,
        moveable: true,
        showCurrentTime: false,
        editable: {
            add: true,         // add new items by double tapping
            updateTime: false,  // drag items horizontally
            updateGroup: false, // drag items from one group to another
            remove: false,       // delete an item by tapping the delete button top right
            overrideItems: false  // allow these options to override item.editable
        },
        onAdd: overview_on_add_callback,
        onUpdate: overview_on_update_callback,
        onMove: on_item_move_overview_callback
    };

    /**
     * @type {string}
     */
    _activePlanetName;

    _userPosition = {
        lat: 0,
        long: 0,
        height: 0,
    };

    /**
     * @type {HTMLElement}
     */
    _timeSpeedSlider;

    _timeSpeedSteps = {
        pause: 0,
        secForward: 1,
        minForward: 2,
        hourForward: 3,
        dayForward: 4,
        monthForward: 5,
        secBack: -1,
        minBack: -2,
        hourBack: -3,
        dayBack: -4,
        monthBack: -5,
    };

    _click = false;
    _currentSpeed;

    /**
     * @type {Date}
     */
    _centerTime;
    _mouseOnTimelineDown;

    _mouseDownLeftTime;
    _minWidth;

    _borderWidth;

    _pause;

    init() {
        this._buttonTemplate = document.getElementById('button-template').content.cloneNode(true);
        this._buttonContainer = document.getElementById('plugin-buttons');

        this._initTimeSpeedSlider();

        this._items = new vis.DataSet();
        this._itemsOverview = new vis.DataSet();

        this._initTimelines();
        this._initEventListener();
    }

    /**
     *
     * @param event {MouseEvent|WheelEvent}
     * @private
     */
    _changeTime(event) {
        if (typeof event.target.dataset.diff === "undefined") {
            return;
        }

        let diff = parseInt(event.target.dataset.diff);

        const date = new Date(this._centerTime.getTime());
        this._centerTime.setSeconds(diff);
        const dif = centerTime.getTime() - date.getTime();
        const hoursDiff = dif / 1000 / 60 / 60;
        CosmoScout.callNative("add_hours_without_animation", hoursDiff);
    }

    /**
     *
     * @param event {WheelEvent}
     * @private
     */
    _changeTimeScroll(event) {
        if (typeof event.target.dataset.diff === "undefined") {
            return;
        }

        let diff = parseInt(event.target.dataset.diff);
        // Data attribute is set in seconds. Call native wants hours
        diff = Math.abs(diff) / 3600;

        if (event.deltaY < 0) {
            diff = -diff
        }

        CosmoScout.callNative('add_hours_without_animation', diff);
    }

    _initEventListener() {
        this._timelineContainer.addEventListener("wheel", this._manualZoomTimeline, true);

        const buttons = [
            "decrease-year-button",
            "decrease-month-button",
            "decrease-day-button",
            "decrease-hour-button",
            "decrease-minute-button",
            "decrease-second-button",

            "increase-year-button",
            "increase-month-button",
            "increase-day-button",
            "increase-hour-button",
            "increase-minute-button",
            "increase-second-button",
        ];

        buttons.forEach(button => {
            const ele = document.getElementById(button);

            if (ele instanceof HTMLElement) {
                ele.addEventListener('click', this._changeTime);
                ele.addEventListener('wheel', this._changeTimeScroll);
            }
        });


        document.getElementById("pause-button").addEventListener('click', toggle_pause);
        document.getElementById("speed-decrease-button").addEventListener('click', decrease_speed);
        document.getElementById("speed-increase-button").addEventListener('click', increase_speed);

        document.getElementById("event-tooltip-location").addEventListener('click', travel_to_item_location);

        document.getElementById("time-reset-button").addEventListener('click', reset_time);

        document.getElementsByClassName('range-label')[0].addEventListener('mousedown', this._rangeUpdateCallback);


        document.getElementById("event-dialog-cancel-button").addEventListener('click', close_form);
        document.getElementById("event-dialog-apply-button").addEventListener('click', apply_event);


        document.getElementById("event-tooltip-container").addEventListener('mouseleave', this._leaveCustomTooltip);
    }

    _initTimeSpeedSlider() {
        this._timeSpeedSlider = document.getElementById('range');

        noUiSlider.create(this._timeSpeedSlider, {
            range: {
                'min': this._timeSpeedSteps.monthBack,
                '4.5%': this._timeSpeedSteps.dayBack,
                '9%': this._timeSpeedSteps.hourBack,
                '13.5%': this._timeSpeedSteps.minBack,
                '18%': this._timeSpeedSteps.secBack,
                '82%': this._timeSpeedSteps.secForward,
                '86.5%': this._timeSpeedSteps.minForward,
                '91%': this._timeSpeedSteps.hourForward,
                '95.5%': this._timeSpeedSteps.dayForward,
                'max': this._timeSpeedSteps.monthForward,
            },
            snap: true,
            start: 1
        });

        this._timeSpeedSlider.noUiSlider.on('update', this._rangeUpdateCallback);
    }

    _firstSliderValue;

    _rangeUpdateCallback() {
        this._currentSpeed = this._timeSpeedSlider.noUiSlider.get();
        if (this._firstSliderValue) {
            document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_right</i>';
            this._firstSliderValue = false;
            return;
        }

        document.getElementById("pause-button").innerHTML = '<i class="material-icons">pause</i>';
        this._timeline.setOptions(this._playingOptions);
        timelineZoomBlocked = true;
        if (parseInt(this._currentSpeed) < this._pause) {
            document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_left</i>';
        } else {
            document.getElementsByClassName("range-label")[0].innerHTML = '<i class="material-icons">chevron_right</i>';
        }

        // 43800 = Month Speed
        this._moveWindow(43800);

        switch (parseInt(this._currentSpeed)) {
            case this._timeSpeedSteps.monthBack:
                CosmoScout.callNative("set_time_speed", -this.MONTHS);
                break;
            case this._timeSpeedSteps.dayBack:
                CosmoScout.callNative("set_time_speed", -this.DAYS);
                break;
            case this._timeSpeedSteps.hourBack:
                CosmoScout.callNative("set_time_speed", -this.HOURS);
                break;
            case this._timeSpeedSteps.minBack:
                CosmoScout.callNative("set_time_speed", -this.MINUTES);
                break;
            case this._timeSpeedSteps.secBack:
                CosmoScout.callNative("set_time_speed", -1);
                break;
            case this._timeSpeedSteps.secForward:
                CosmoScout.callNative("set_time_speed", 1);
                break;
            case this._timeSpeedSteps.minForward:
                CosmoScout.callNative("set_time_speed", this.MINUTES);
                break;
            case this._timeSpeedSteps.hourForward:
                CosmoScout.callNative("set_time_speed", this.HOURS);
                break;
            case this._timeSpeedSteps.dayForward:
                CosmoScout.callNative("set_time_speed", this.DAYS);
                break;
            case this._timeSpeedSteps.monthForward:
                CosmoScout.callNative("set_time_speed", this.MONTHS);
                break;
            default:
        }
    }


    _rightTimeId = 'rightTime';

    _leftTimeId = 'leftTime';

    _timeId = 'custom';

    _initTimelines() {
        const timelineContainer = document.getElementById('timeline');
        timelineContainer.addEventListener("wheel", this._manualZoomTimeline, true);

        const overviewContainer = document.getElementById('overview');

        this._timeline = new vis.Timeline(timelineContainer, this._items, this._timelineOptions);
        this._centerTime = this._timeline.getCurrentTime();
        this._timeline.on('select', this._onSelect);
        this._timeline.moveTo(this._centerTime, {
            animation: false,
        });

        this._timeline.addCustomTime(this._this._centerTime, this._timeId);
        this._timeline.on('click', this._onClickCallback);
        this._timeline.on('changed', this._timelineChangedCallback);
        this._timeline.on('mouseDown', this._mouseDownCallback);
        this._timeline.on('mouseUp', this._mouseUpCallback);
        this._timeline.on('rangechange', this._rangeChangeCallback);
        this._timeline.on('itemover', this._itemOverCallback);
        this._timeline.on('itemout', this._itemOutCallback);

//create overview timeline
        this._overviewTimeline = new vis.Timeline(overviewContainer, this._itemsOverview, this._overviewTimelineOptions);
        this._overviewTimeline.addCustomTime(this._timeline.getWindow().end, this._rightTimeId);
        this._overviewTimeline.addCustomTime(this._timeline.getWindow().start, this._leftTimeId);
        this._overviewTimeline.on('select', this._onSelect);
        this._overviewTimeline.on('click', this._onClickCallback);
        this._overviewTimeline.on('changed', this._drawFocusLens);
        this._overviewTimeline.on('mouseDown', this._overviewMouseDownCallback);
        this._overviewTimeline.on('rangechange', this._overviewChangedCallback);
        this._overviewTimeline.on('itemover', this._itemOverOverviewCallback);
        this._overviewTimeline.on('itemout', this._itemOutCallback);
        this._initialOverviewWindow(new Date(1950, 1), new Date(2030, 12));
    }

    _initialOverviewWindow(start, end) {
        this._overviewTimeline.setWindow(start, end, {
            animations: false,
        });
    }

    _tooltipVisible;

    /**
     *
     * @param properties {VisTimelineEvent}
     * @private
     */
    _itemOutCallback(properties) {
        if (properties.event.toElement.className !== "event-tooltip") {
            document.getElementById("event-tooltip-container").style.display = "none";
            this._tooltipVisible = false;
            hoveredHTMLEvent.classList.remove('mouseOver');
        }
    }

    _itemOverOverviewCallback(properties) {
        this._itemOverCallback(properties, true);
    }

    _moveWindow() {
        const step = convert_seconds(timelineRangeFactor);
        let startDate = new Date(centerTime.getTime());
        let endDate = new Date(centerTime.getTime());
        startDate = DateOperations.decreaseDate(startDate, step.days, step.hours, step.minutes, step.seconds, step.milliSec);
        endDate = DateOperations.increaseDate(endDate, step.days, step.hours, step.minutes, step.seconds, step.milliSec);
        this._timeline.setWindow(startDate, endDate, {
            animations: false,
        });
    }

    _hoveredHTMLEvent;

    /**
     *
     * @param properties {VisTimelineEvent}
     * @param overview
     * @private
     */
    _itemOverCallback(properties, overview) {
        document.getElementById("event-tooltip-container").style.display = "block";
        this._tooltipVisible = true;
        for (const item in this._items._data) {
            if (this._items._data[item].id === properties.item) {
                document.getElementById("event-tooltip-content").innerHTML = this._items._data[item].content;
                document.getElementById("event-tooltip-description").innerHTML = this._items._data[item].description;
                document.getElementById("event-tooltip-location").innerHTML = "<i class='material-icons'>send</i> " + this._items._data[item].planet + " " + this._items._data[item].place;
                hoveredItem = this._items._data[item];
            }
        }
        const events = document.getElementsByClassName(properties.item);
        let event;
        for (let i = 0; i < events.length; i++) {
            if (!overview && $(events[i]).hasClass("event")) {
                event = events[i];
            } else if (overview && $(events[i]).hasClass("overviewEvent")) {
                event = events[i];
            }
        }
        this._hoveredHTMLEvent = event;
        this._hoveredHTMLEvent.classList.add('mouseOver');
        const eventRect = event.getBoundingClientRect();
        const left = eventRect.left - 150 < 0 ? 0 : eventRect.left - 150;
        document.getElementById("event-tooltip-container").style.top = eventRect.bottom + 'px';
        document.getElementById("event-tooltip-container").style.left = left + 'px';
        if (this._currentSpeed !== this._pause) {
            start_redraw_tooltip(event);
        }
    }

    _leaveCustomTooltip() {
        document.getElementById("event-tooltip-container").style.display = "none";
        this._tooltipVisible = false;
        this._hoveredHTMLEvent.classList.remove('mouseOver');
    }


    _mouseDownCallback() {
        this._timeline.setOptions(this._pauseOptions);
        this._mouseOnTimelineDown = true;
        this._lastPlayValue = this._currentSpeed;
        this._click = true;
        this._mouseDownLeftTime = this._timeline.getWindow().start;
    }

    _lastPlayValue;

    _mouseUpCallback() {
        if (this._mouseOnTimelineDown && this._lastPlayValue !== this._pause) {
            this._timeSpeedSlider.noUiSlider.set(parseInt(this._lastPlayValue));
        }
        this._mouseOnTimelineDown = false;
    }

    _overviewMouseDownCallback() {
        this._click = true;

    }

    _overviewChangedCallback() {
        this._click = false;
    }

    _timelineChangedCallback() {
        this._setOverviewTimes();
        this._drawFocusLens();
    }

    _rangeChangeCallback(properties) {
        if (properties.byUser && !properties.event instanceof WheelEvent) {
            if (this._currentSpeed !== this._pause) {
                set_pause();
            }
            click = false;
            const dif = properties.start.getTime() - this._mouseDownLeftTime.getTime();
            const secondsDif = dif / 1000;
            const hoursDif = secondsDif / 60 / 60;

            const step = convert_seconds(secondsDif);
            let date = new Date(this._centerTime.getTime());
            date = increase_date(date, step.days, step.hours, step.minutes, step.seconds, step.milliSec);
            set_date_local(date);
            this._mouseDownLeftTime = new Date(properties.start.getTime());
            window.call_native("add_hours_without_animation", hoursDif);
        }
    }

    _setOverviewTimes() {
        this._overviewTimeline.setCustomTime(timeline.getWindow().end, this._rightTimeId);
        this._overviewTimeline.setCustomTime(timeline.getWindow().start, this._leftTimeId);
        this._drawFocusLens();
    }


    _drawFocusLens() {
        const leftCustomTime = document.getElementsByClassName("leftTime")[0];
        const leftRect = leftCustomTime.getBoundingClientRect();
        const rightCustomTime = document.getElementsByClassName("rightTime")[0];
        const rightRect = rightCustomTime.getBoundingClientRect();

        let divElement = document.getElementById("focus-lens");
        divElement.style.position = "absolute";
        divElement.style.left = leftRect.right + 'px';
        divElement.style.top = (leftRect.top + offset) + 'px';

        const height = leftRect.bottom - leftRect.top - shorten;
        let width = rightRect.right - leftRect.left;

        let xValue = 0;
        if (width < this._minWidth) {
            width = this._minWidth + 2 * this._borderWidth;
            xValue = -(leftRect.left + this._minWidth - rightRect.right) / 2 - this._borderWidth;
            xValue = Math.round(xValue);
            divElement.style.transform = " translate(" + xValue + "px, 0px)";
        } else {
            divElement.style.transform = " translate(0px, 0px)";
        }

        divElement.style.height = height + 'px';
        divElement.style.width = width + 'px';

        divElement = document.getElementById("focus-lens-left");
        divElement.style.top = (leftRect.top + offset + height) + 'px';
        width = leftRect.right + xValue + this._borderWidth;
        width = width < 0 ? 0 : width;
        divElement.style.width = width + 'px';
        const body = document.getElementsByTagName("body")[0];
        let bodyRect = body.getBoundingClientRect();

        divElement = document.getElementById("focus-lens-right");
        divElement.style.top = (leftRect.top + offset + height) + 'px';
        width = bodyRect.right - rightRect.right + xValue + 1;
        width = width < 0 ? 0 : width;
        divElement.style.width = width + 'px';
    }

    _onClickCallback(properties) {
        if (this._click) {
            this._generalOnClick(properties)
        }
    }

    /**
     * Changes the size of the displayed time range while the simulation is still playing
     *
     * @param event
     * @private
     */
    _manualZoomTimeline(event) {
        if (timelineZoomBlocked) {
            if (event.deltaY < 0) {
                timelineRangeFactor -= timelineRangeFactor * zoomPercentage;
                if (timelineRangeFactor < minRangeFactor) {
                    timelineRangeFactor = minRangeFactor;
                }
            } else {
                timelineRangeFactor += timelineRangeFactor * zoomPercentage;
                if (timelineRangeFactor > maxRangeFactor) {
                    timelineRangeFactor = maxRangeFactor;
                }
            }
            this._rangeUpdateCallback();
        }
    }

    /**
     * Vis Timeline Event Properties
     * https://visjs.github.io/vis-timeline/docs/timeline/#getEventProperties
     *
     * @param properties {VisTimelineEvent}
     * @private
     */
    _onSelect(properties) {
        for (const item in this._items._data) {
            if (this._items._data[item].id === properties.items) {
                let dif = this._items._data[item].start.getTime() - this._centerTime.getTime();
                let hoursDif = dif / 1000 / 60 / 60;

                if (this._items._data[item].start.getTimezoneOffset() > this._centerTime.getTimezoneOffset()) {
                    hoursDif -= 1;
                } else if (this._items._data[item].start.getTimezoneOffset() < this._centerTime.getTimezoneOffset()) {
                    hoursDif += 1;
                }

                window.call_native("add_hours", hoursDif);
                travel_to(true, this._items._data[item].planet, this._items._data[item].place, this._items._data[item].content);
            }
        }
    }

    /**
     * Vis Timeline Event Properties
     * https://visjs.github.io/vis-timeline/docs/timeline/#getEventProperties
     *
     * @param properties {VisTimelineEvent}
     * @private
     */
    _generalOnClick(properties) {
        if (properties.what !== "item" && properties.time != null) {
            const dif = properties.time.getTime() - this._centerTime.getTime();
            let hoursDif = dif / 1000 / 60 / 60;

            if (properties.time.getTimezoneOffset() > this._centerTime.getTimezoneOffset()) {
                hoursDif -= 1;
            } else if (properties.time.getTimezoneOffset() < this._centerTime.getTimezoneOffset()) {
                hoursDif += 1;
            }
            window.call_native("add_hours", hoursDif);
        }
    }

    /**
     * Adds a button to the button bar
     *
     * @param icon {string} Materialize icon name
     * @param tooltip {string} Tooltip text that gets shown if the button is hovered
     * @param callback {string} Function name passed to call_native
     */
    addButton(icon, tooltip, callback) {
        let button = this._buttonTemplate.cloneNode(true).firstElementChild;

        button.innerHTML = button.innerHTML
            .replace('%ICON%', icon)
            .trim();

        button.setAttribute('title', tooltip);

        button.addEventListener('click', () => {
            CosmoScout.callNative(callback);
        });

        this._buttonContainer.appendChild(button);

        CosmoScout.initTooltips();
    }

    setActivePlanet(name) {
        this._activePlanetName = name;
    }

    /**
     *
     * @param long {number}
     * @param lat {number}
     * @param height {number}
     */
    setUserPosition(long, lat, height) {
        this._userPosition = {
            long,
            lat,
            height
        }
    }

    /**
     * Rotates the button bar compass
     *
     * @param angle {number}
     */
    setNorthDirection(angle) {
        document.getElementById('compass-arrow').style.transform = `rotateZ(${angle}rad)`;
    }

    setDate(date) {

    }

    addItem(start, end, id, content, style, description, planet, place) {
        const data = {};
        data.start = new Date(start);
        data.id = id;
        if (end !== "") {
            data.end = new Date(end);
        }
        if (style !== "") {
            data.style = style;
        }
        data.planet = planet;
        data.description = description;
        data.place = place;
        data.content = content;
        data.className = 'event ' + id;
        items.update(data);
        data.className = 'overviewEvent ' + id;
        itemsOverview.update(data);
    }

    /**
     * Called from Application.cpp 622/816
     *
     * @param speed {number}
     */
    setTimeSpeed(speed) {
        let notification = [];

        switch (speed) {
            case this.PAUSE:
                set_pause();
                notification = ['Pause', 'Time is paused.', 'pause'];
                break;

            case this.REALTIME:
                notification = ["Speed: Realtime", "Time runs in realtime.", "play_arrow"];
                break;

            case this.MINUTES:
                notification = ["Speed: Min/s", "Time runs at one minute per second.", "fast_forward"];
                break;

            case this.HOURS:
                notification = ["Speed: Hour/s", "Time runs at one hour per second.", "fast_forward"];
                break;

            case this.DAYS:
                notification = ["Speed: Day/s", "Time runs at one day per second.", "fast_forward"];
                break;

            case this.MONTHS:
                notification = ["Speed: Month/s", "Time runs at one month per second.", "fast_forward"];
                break;

            /* Negative times */
            case -this.REALTIME:
                notification = ["Speed: -Realtime", "Time runs backwards in realtime.", "fast_rewind"];
                break;

            case -this.MINUTES:
                notification = ["Speed: -Min/s", "Time runs backwards at one minute per second.", "fast_rewind"];
                break;

            case -this.HOURS:
                notification = ["Speed: -Hour/s", "Time runs backwards at one hour per second.", "fast_rewind"];
                break;

            case -this.DAYS:
                notification = ["Speed: -Day/s", "Time runs backwards at one day per second.", "fast_rewind"];
                break;

            case -this.MONTHS:
                notification = ["Speed: -Month/s", "Time runs backwards at one month per second.", "fast_rewind"];
                break;
        }

        if (notification.length > 0) {
            CosmoScout.call('notifications', 'printNotification', ...notification);
        }
    }

    /**
     *
     * @param direct {boolean}
     * @param planet {string}
     * @param place {string} Location string in the form of '3.635° E 26.133° S 10.0 Tsd km'
     * @param name {string}
     */
    travelTo(direct, planet, place, name) {
        let placeArr = place.split(' ');

        let animationTime = direct ? 0 : 5;

        CosmoScout.call(
            'flyto',
            'flyTo',
            planet,
            this._parseLongitude(placeArr[0], placeArr[1]),
            this._parseLatitude(placeArr[2], placeArr[3]),
            this._parseHeight(placeArr[4], placeArr[5]),
            animationTime
        );
    }

    /**
     * Parses a latitude string for travelTo
     *
     * @see {travelTo}
     * @param lat {string}
     * @param half {string}
     * @return {number}
     * @private
     */
    _parseLatitude(lat, half) {
        let latitude = parseFloat(lat);
        if (half === 'S') {
            latitude = -latitude;
        }

        return latitude;
    }

    /**
     * Parses a longitude string for travelTo
     *
     * @see {travelTo}
     * @param lon {string}
     * @param half {string}
     * @return {number}
     * @private
     */
    _parseLongitude(lon, half) {
        let longitude = parseFloat(lon);
        if (half === 'W') {
            longitude = -longitude;
        }

        return longitude;
    }

    /**
     * Parses a height string
     *
     * @param heightStr {string}
     * @param unit {string}
     * @return {number}
     * @private
     */
    _parseHeight(heightStr, unit) {
        const height = parseFloat(heightStr);

        switch (unit) {
            case 'mm':
                return height / 1000;
            case 'cm':
                return height / 100;
            case 'm':
                return height;
            case 'km':
                return height * 1e3;
            case 'Tsd':
                return height * 1e6;
            case 'AU':
                return height * 1.496e11;
            case 'ly':
                return height * 9.461e15;
            case 'pc':
                return height * 3.086e16;
            default:
                return height * 3.086e19;
        }
    }
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
     * Initialize all containers
     *
     * @param config {{userPosition: string, pointerPosition: string, speed: string}}
     */
    init(config = {}) {
        Object.assign(this.config, config);

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
        if (typeof location === "undefined") {
            CosmoScout.callNative('fly_to', planet);
        } else {
            CosmoScout.callNative('fly_to', planet, location.longitude, location.latitude, location.height, time);
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