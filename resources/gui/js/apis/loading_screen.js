/* global IApi, CosmoScout */

/**
 * The loading screen
 */
class LoadingScreenApi extends IApi {
  /**
   * @inheritDoc
   */
  name = 'loadingScreen';

  /**
   * @type {HTMLElement}
   */
  _loadingScreen;

  /**
   * @type {HTMLElement}
   */
  _version;

  /**
   * @type {HTMLElement}
   */
  _status;

  /**
   * @type {HTMLElement}
   */
  _progressBar;

  /**
   * @inheritDoc
   */
  init() {
    this._loadingScreen = document.getElementById('loading-screen');
    this._version = document.getElementById('version');
    this._status = document.getElementById('lower');
    this._progressBar = document.getElementById('progress-bar');
  }

  /**
   * Enable the loading screen.
   * Removes dependencies if enable is false
   *
   * @param {boolean} enable
   */
  setLoading(enable) {
    if (enable) {
      document.body.classList.add('loading');
    } else {
      document.body.classList.remove(...document.body.classList);
      document.body.classList.add('loaded');

      setTimeout(() => {
        document.body.removeChild(this._loadingScreen);
        CosmoScout.gui.unregisterCss('css/loading_screen.css');
        CosmoScout.gui.unregisterJavaScript('js/apis/loading_screen.js');
        CosmoScout.removeApi('loading_screen');
      }, 1500);
    }
  }

  /**
   * Sets the status text like current plugin
   *
   * @param {string} text
   */
  setStatus(text) {
    this._status.innerHTML = text;
  }

  /**
   * Updates the progress bar progress
   *
   * @param {number} percent
   * @param {boolean} animate
   */
  setProgress(percent, animate) {
    if (animate) {
      this._progressBar.classList.add('animated');
    } else {
      this._progressBar.classList.remove('animated');
    }

    this._progressBar.style.width = `${percent}%`;
  }

  /**
   * Sets the version string
   *
   * @param {string} text
   */
  setVersion(text) {
    this._version.innerHTML = text;
  }
}
