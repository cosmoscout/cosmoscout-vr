/* global IApi, CosmoScout */

(() => {
  /**
   * UserStudy Api
   */
  class UserStudyApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'userStudy';

    /**
     * @inheritDoc
     */
    init() {
      CosmoScout.gui.initSlider("userStudy.setRecordingInterval", 1, 20, 1, [5]);
    }
  }

  CosmoScout.init(UserStudyApi);
})();
