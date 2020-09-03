/* global IApi, CosmoScout */

(() => {
  /**
   * Sharad Api
   */
  class SharadApi extends IApi {
    /**
     * @inheritDoc
     */
    name = 'sharad';

    /**
     * @param file {string}
     * @param time {string|number}
     */
    add(file, time) {
      const sharadList = document.getElementById('list-sharad');
      const sharad     = CosmoScout.gui.loadTemplateContent('sharad');

      sharad.innerHTML = sharad.innerHTML.replace(/%FILE%/g, file).replace(/%TIME%/g, time).trim();

      sharad.classList.add(`item-${file}`);

      sharadList.appendChild(sharad);
    }
  }

  CosmoScout.init(SharadApi);
})();
