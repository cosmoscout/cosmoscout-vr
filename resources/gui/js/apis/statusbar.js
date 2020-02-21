/* global IApi, Format, CosmoScout */

/**
 * Statusbar Api
 */
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
      userPosition: 'statusbar-user-position',
      pointerPosition: 'statusbar-pointer-position',
      speed: 'statusbar-speed',
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
     * @type {number}
     * @private
     */
    _observerSpeed = 0.0;

    /**
     * @type {array}
     * @private
     */
    _observerPosition = [0.0, 0.0, 0.0];

    /**
     * @type {array}
     * @private
     */
    _pointerPosition = null;

    /**
     * @type {string}
     * @private
     */
    _activePlanetCenter = "";

    /**
     * @type {string}
     * @private
     */
    _activePlanetFrame = "";

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
     * Called regularly by CosmoScout VR
     *
     * @param speed {number}
     */
    setObserverSpeed(speed) {
      this._speedContainer.innerText = Format.speed(speed);
      this._observerSpeed = speed;
    }
  
    getObserverSpeed() {
      return this._observerSpeed;
    }
    
    /**
     * Called regularly by CosmoScout VR
     *
     * @param long {number}
     * @param lat {number}
     * @param height {number}
     */
    setObserverPosition(lat, lng, height) {
      this._userContainer.innerText = `${Format.longitude(lng) + Format.latitude(lat)}(${Format.height(height)})`;
      this._observerPosition = [lat, lng, height];
    }
    
    /**
     * @return The current position of the observer in [lng, lat, height]
     */
    getObserverPosition() {
      return this._observerPosition;
    }
    
    /**
     * Called regularly by CosmoScout VR
     *
     * @param hits {bool}
     * @param long {number}
     * @param lat {number}
     * @param height {number}
     */
    setPointerPosition(hits, lat, lng, height) {
      if (hits) {
        this._pointerContainer.innerText = `${Format.longitude(lng) + Format.latitude(lat)}(${Format.height(height)})`;
        this._pointerPosition = [lat, lng, height];
      } else {
        this._pointerContainer.innerText = ' - ';
        this._pointerPosition = null;
      }
    }
    
    /**
     * @return The current position of the mouse pointer in [lng, lat, height] - may be null {array}
     */
    getPointerPosition() {
      return _pointerPosition;
    }

    /**
     * Called regularly by CosmoScout VR
     *
     * @param centerName {string}
     */
    setActivePlanetCenter(centerName) {
      this._activePlanetCenter = centerName;
    }
    
    /**
     * @return The current planet's center name {string}
     */
    getActivePlanetCenter() {
      return this._activePlanetCenter;
    }

    /**
     * Called regularly by CosmoScout VR
     *
     * @param frameName {string}
     */
    setActivePlanetFrame(frameName) {
      this._activePlanetFrame = frameName;
    }
    
    /**
     * @return The current planet's frame name {string}
     */
    getActivePlanetframe() {
      return this._activePlanetFrame;
    }
}
