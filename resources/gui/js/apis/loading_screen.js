class LoadingScreenApi extends IApi {
  name = 'loading_screen';

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

  init() {
    this._loadingScreen = document.getElementById('loading-screen');
    this._version = document.getElementById('version');
    this._status = document.getElementById('lower');
    this._progressBar = document.getElementById('progress-bar');
  }

  setLoading(enable) {
    if (enable) {
      document.body.classList.add('loading');
    } else {
      document.body.classList.remove(...document.body.classList);
      document.body.classList.add('loaded');

      setTimeout(() => {
        document.body.removeChild(this._loadingScreen);
        CosmoScout.unregisterCss('css/loading_screen.css');
        CosmoScout.unregisterJavaScript('js/apis/loading_screen.js');
        CosmoScout.remove('loading_screen');
      }, 1500);
    }
  }

  setStatus(text) {
    this._status.innerHTML = text;
  }

  setProgress(percent, animate) {
    if (animate) {
      this._progressBar.classList.add('animated');
    } else {
      this._progressBar.classList.remove('animated');
    }

    this._progressBar.style.width = `${percent}%`;
  }

  setVersion(text) {
    this._version.innerHTML = text;
  }
}
