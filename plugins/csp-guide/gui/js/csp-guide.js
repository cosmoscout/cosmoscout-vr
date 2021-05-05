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

    agent = null;

    #clippyApi = null;
    #lastState = null;

    #clippyWidth  = 0;
    #clippyHeight = 0;

    /**
     * @inheritDoc
     */
    init() {
      import("./third-party/js/clippyjs/index.js").then(m => {
        this.#clippyApi = m.default; new this.#clippyApi.load(
            'Clippy', (agent) => { this.agent = agent;}, undefined, './third-party/agents/');
      });

      this.#lastState = deepCopy(CosmoScout.state);
    }

    update() {
      if (this.agent !== null) {
        this.#greet();

        if (this.#hasGreetedFinished) {
          this.#checkCenter();
          this.#planetRotateChecker();
        }
      }

      this.#lastState = deepCopy(CosmoScout.state);
    }

    #hasGreeted         = false;
    #hasGreetedFinished = false;
    #greet() {
      if (!this.#hasGreeted) {
        this.#hasGreeted = true;
        this.agent.show();
        this.agent.play("Greeting");
        this.agent.speak(
            "Hi, I am your guide, Clippy. I am here to help you navigate this large Universe.");
        this.agent.play("IdleHeadScratch");
        this.agent.speak("I will also help you navigate the user interface, of course.");
        this.agent.play("Wave");

        this.agent.onEmptyQueue(() => {
          this.#hasGreetedFinished = true;
          this.#clippyWidth  = document.querySelector(".clippy").getBoundingClientRect().width;
          this.#clippyHeight = document.querySelector(".clippy").getBoundingClientRect().height;
        });
      }
    }

    #centerChangeDisplayed = false;
    #checkCenter() {
      if (CosmoScout.flyToLocations) {
        const prevCenter = this.#lastState.activePlanetCenter;
        const currCenter = CosmoScout.state.activePlanetCenter;

        if (!this.#centerChangeDisplayed && prevCenter !== currCenter) {
          this.#centerChangeDisplayed = true;

          const bookmarksTabElement = document.querySelector("#heading-sidebar-tab-Bookmarks");
          const bookmarksRect       = bookmarksTabElement.getBoundingClientRect();

          this.agent.play("GetAttention");
          this.agent.moveTo(bookmarksRect.right + 50,
              (bookmarksRect.top + bookmarksRect.bottom - this.#clippyHeight) / 2);
          this.agent.gestureAt(bookmarksRect.left,
              (bookmarksRect.top + bookmarksRect.bottom - this.#clippyHeight) / 2);
          this.agent.speak(
              "Hi, I see you moved away from a planet. You can use the bookmarks tab to move to places. This way you won't get lost in the infinite space of the Universe!");
          this.agent.gestureAt(bookmarksRect.left,
              (bookmarksRect.top + bookmarksRect.bottom - this.#clippyHeight) / 2);
          this.agent.moveTo(bookmarksRect.right + 350, bookmarksRect.top);

          this.agent.onEmptyQueue(() => {
            bookmarksTabElement.click();
            this.agent.speak("Try clicking on a planet!");
            this.agent.play("LookDownRight");
          });
        }
      }
    }

    #planetRotated = false;
    #planetRotateChecker() {
      function length2(vector) {
        return vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2;
      }

      function isSame(a, b) {
        return a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
      }

      const prevPos = this.#lastState.observerPosition;
      const currPos = CosmoScout.state.observerPosition;

      if (!this.#planetRotated && !isSame(prevPos, currPos) &&
          length2(prevPos) !== length2(currPos)) {
        this.#planetRotated = true;

        this.agent.play("GetAttention");
        this.agent.speak(
            "I see you rotated the planet! You can interact in different ways with the planet.");
        {
          const turnNorthUpButton     = document.querySelector("#turn-north-up-button");
          const turnNorthUpButtonRect = turnNorthUpButton.getBoundingClientRect();
          this.agent.moveTo(turnNorthUpButtonRect.left, turnNorthUpButtonRect.bottom);
          this.agent.play("LookUpRight");
          this.agent.speak(
              "You can click the button with the compass to orient the camera so north is the up direction.");
        }
        {
          const fixHorizonButton     = document.querySelector("#fix-horizon-button");
          const fixHorizonButtonRect = fixHorizonButton.getBoundingClientRect();
          this.agent.moveTo(
              (fixHorizonButtonRect.left + fixHorizonButtonRect.right - this.#clippyWidth) / 2,
              fixHorizonButtonRect.bottom);
          this.agent.play("GestureUp");
          this.agent.speak("You can click this button to look at the horizon.");
        }
        {
          const landOnSurfaceButton     = document.querySelector("#land-on-surface-button");
          const landOnSurfaceButtonRect = landOnSurfaceButton.getBoundingClientRect();
          this.agent.moveTo(
              (landOnSurfaceButtonRect.left + landOnSurfaceButtonRect.right - this.#clippyWidth) /
                  2,
              landOnSurfaceButtonRect.bottom);
          this.agent.play("GestureUp");
          this.agent.speak("You can click this button to land on the surface.");
        }
        {
          const launchToOrbitButton     = document.querySelector("#launch-to-orbit-button");
          const launchToOrbitButtonRect = launchToOrbitButton.getBoundingClientRect();
          this.agent.moveTo(
              (launchToOrbitButtonRect.left + launchToOrbitButtonRect.right - this.#clippyWidth) /
                  2,
              launchToOrbitButtonRect.bottom);
          this.agent.play("GestureUp");
          this.agent.speak("You can click this button to fly back into orbit.");
        }

        this.agent.play("Explain");
      }
    }
  }

  CosmoScout.init(GuideApi);

  function deepCopy(object) {
    return JSON.parse(JSON.stringify(object));
  }
})();

//# sourceURL=csp-guide.js