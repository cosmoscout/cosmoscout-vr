// API calls -------------------------------------------------------------------------------
function set_pointer_position(hits, long, lat, height) {
    return CosmoScout.call('statusbar', 'setPointerPosition', hits, long, lat, height);
}

function set_user_position(long, lat, height) {
    return CosmoScout.call('statusbar', 'setUserPosition', long, lat, height);
}

function set_speed(speed) {
    return CosmoScout.call('statusbar', 'setSpeed', speed);
}