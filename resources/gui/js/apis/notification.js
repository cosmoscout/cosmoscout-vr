/* global IApi, CosmoScout */

/**
 * Notifications
 */
class NotificationApi extends IApi {
  /**
   * @inheritDoc
   */
  name = 'notifications';

  /**
   * @type {HTMLElement}
   * @private
   */
  _container;

  /**
   * Set the container in which to place the notifications
   *
   * @param container {string}
   */
  init(container = 'notifications') {
    this._container = document.getElementById(container);

    if (this._container === null) {
      console.error(`Element with id #${container} not found.`);
    }
  }

  /**
   * Adds a notification into the initialized notification container
   *
   * @param title {string} Title
   * @param content {string} Content
   * @param icon {string} Materialize Icon Name
   * @param flyTo {string} Optional flyto name which gets passed to 'fly_to'. Activated on click
   */
  print(title, content, icon, flyTo) {
    if (this._container === null) {
      console.error('Notification container is not defined! Did you call "init"?');
      return;
    }

    if (this._container.children.length > 4) {
      const no = this._container.lastElementChild;

      clearTimeout(no.timer);

      this._container.removeChild(no);
    }

    let notification;
    try {
      notification = this._makeNotification(title, content, icon);
    } catch (e) {
      console.warn(e);
      return;
    }

    this._container.prepend(notification);

    if (flyTo) {
      notification.classList.add('clickable');
      notification.addEventListener('click', () => {
        CosmoScout.callbacks.navigation.flyTo(flyTo);
      });
    }

    notification.timer = setTimeout(() => {
      notification.classList.add('fadeout');
      this._container.removeChild(notification);
    }, 8000);

    setTimeout(() => {
      notification.classList.add('show');
    }, 60);
  }

  /**
   * Creates the actual HTML Notification
   *
   * @param title {string}
   * @param content {string}
   * @param icon {string}
   * @return {HTMLElement}
   * @private
   */
  _makeNotification(title, content, icon = '') {
    const notification = CosmoScout.gui.loadTemplateContent('notification');

    if (notification === false) {
      throw new Error(
          'Notification template content could not be loaded. Does "#notification-template" exist?');
    }

    notification.innerHTML = notification.innerHTML.replace('%TITLE%', title)
                                 .replace('%CONTENT%', content)
                                 .replace('%ICON%', icon)
                                 .trim();

    return notification;
  }
}
