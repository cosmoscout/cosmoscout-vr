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

  update() {
    let pos = CosmoScout.state.pointerPosition;
    if (pos !== undefined) {
      this._pointerContainer.innerText = `${CosmoScout.utils.formatLongitude(pos[0]) + CosmoScout.utils.formatLatitude(pos[1])}(${CosmoScout.utils.formatHeight(pos[2])})`;
    } else {
      this._pointerContainer.innerText = ' - ';
    }
    
    pos = CosmoScout.state.observerPosition;
    if (pos !== undefined) {
      this._userContainer.innerText = `${CosmoScout.utils.formatLongitude(pos[0]) + CosmoScout.utils.formatLatitude(pos[1])}(${CosmoScout.utils.formatHeight(pos[2])})`;
    } else {
      this._userContainer.innerText = ' - ';
    }

    if (CosmoScout.state.observerSpeed !== undefined) {
      this._speedContainer.innerText = CosmoScout.utils.formatSpeed(CosmoScout.state.observerSpeed);
    }
  }
}
