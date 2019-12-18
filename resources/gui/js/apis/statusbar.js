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
      this._userContainer.innerText = `${Format.longitude(long) + Format.latitude(lat)}(${Format.height(height)})`;
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
        text = `${Format.longitude(long) + Format.latitude(lat)}(${Format.height(height)})`;
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
