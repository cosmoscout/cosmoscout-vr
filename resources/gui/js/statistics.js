
function set_data(data, frameRate) {
    return CosmoScout.call('statistics', 'setData', data, frameRate);
}
