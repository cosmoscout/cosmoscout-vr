/* global IApi, CosmoScout */

(() => {
  /**
   * Guide Api
   */
  class GuideApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'guide';

    clippyApi = null;
    agent = null;

    /**
     * @inheritDoc
     */
    init() {
      import("./third-party/js/clippyjs/index.js").then(m => {
        this.clippyApi = m.default;
        new this.clippyApi.load('Clippy', (agent) => {
          this.agent = agent;
          this.agent.show();
          setTimeout(10000, () => {
            this.agent.moveTo(500, 500);
            this.agent.animate();
          })
        });
      });
    }
  }

  CosmoScout.init(GuideApi);
})();

//# sourceURL=csp-guide.js