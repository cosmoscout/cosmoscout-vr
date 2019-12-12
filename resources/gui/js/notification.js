function print_notification(title, content, icon, flyto) {
    return CosmoScout.call('notifications', 'printNotification', title, content, icon, flyto);
}
