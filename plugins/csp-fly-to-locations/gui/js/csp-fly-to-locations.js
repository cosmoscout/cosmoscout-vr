/* global IApi, CosmoScout */

(() => {
  /**
   * FlyTo Api
   */
  class FlyToLocationsApi extends IApi {
    /**
     * @inheritDoc
     * @type {string}
     */
    name = 'flyToLocations';

    /**
     * Adds a bookmark button to the grid of icon bookmarks.
     *
     * @param bookmarkID {number}
     * @param bookmarkName {string}
     * @param bookmarkIcon {string}
     */
    addGridBookmark(bookmarkID, bookmarkName, bookmarkIcon) {
      let button       = CosmoScout.gui.loadTemplateContent('flytolocations-bookmarks-grid-button');
      button.innerHTML = button.innerHTML.replace(/%NAME%/g, bookmarkName)
                             .replace(/%ICON%/g, bookmarkIcon)
                             .replace(/%ID%/g, bookmarkID)
                             .trim();
      button.id = `flytolocations-bookmark-${bookmarkID}`;
      document.getElementById('flytolocations-bookmarks-grid').appendChild(button);

      this._sortBookmarks(document.getElementById('flytolocations-bookmarks-grid'));
    }

    /**
     * Adds a bookmark button to the list of position bookmarks.
     *
     * @param bookmarkID {number}
     * @param bookmarkName {string}
     * @param bookmarkHasTime {boolean}
     */
    addListBookmark(bookmarkID, bookmarkName, bookmarkHasTime) {
      let listItem = CosmoScout.gui.loadTemplateContent('flytolocations-bookmarks-list-item');
      listItem.innerHTML =
          listItem.innerHTML.replace(/%NAME%/g, bookmarkName).replace(/%ID%/g, bookmarkID).trim();
      listItem.id = `flytolocations-bookmark-${bookmarkID}`;

      if (!bookmarkHasTime) {
        listItem.querySelector(".flytolocations-bookmarks-time-button").classList.add("hidden");
      }

      document.getElementById('flytolocations-bookmarks-list').appendChild(listItem);

      CosmoScout.gui.initTooltips();

      this._sortBookmarks(document.getElementById('flytolocations-bookmarks-list'));
    }

    /**
     * Removes a bookmark by ID. It will be removed regardless in which bookmark areas it is.
     *
     * @param group {string}
     * @param text {string}
     */
    removeBookmark(bookmarkID) {
      let bookmark = document.querySelector("#flytolocations-bookmark-" + bookmarkID);
      if (bookmark) {
        bookmark.remove();
      }
    }

    /**
     * Sorts the grid or list bookmarks alphabetically.
     *
     * @param container {HTMLElement} The element containing bookmark divs.
     * @private
     */
    _sortBookmarks(container) {
      Array.prototype.slice.call(container.children)
          .sort((ea, eb) => {
            let a = ea.querySelector(".flytolocations-bookmarks-name").textContent;
            let b = eb.querySelector(".flytolocations-bookmarks-name").textContent;
            return a < b ? -1 : (a > b ? 1 : 0);
          })
          .forEach((div) => {
            container.appendChild(div);
          });
    }
  }

  CosmoScout.init(FlyToLocationsApi);
})();
