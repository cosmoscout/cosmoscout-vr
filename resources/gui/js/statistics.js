var values = [];
var colorHash = new ColorHash({lightness: 0.5, saturation: 0.3});

function set_data(data, frameRate) {
    data = JSON.parse(data);

    // first set all times to zero
    values.forEach(function (element) {
        if (data[element.name]) {
            element.timeGPU = data[element.name][0];
            element.timeCPU = data[element.name][1];
            data[element.name][0] = -1;
            data[element.name][1] = -1;
        } else {
            element.timeGPU = 0;
            element.timeCPU = 0;
        }
    });

    // then add all new elements
    for (var key in data) {
        if (!data.hasOwnProperty(key)) continue;

        if (data[key][0] >= 0) {
            values.push({
                "name": key,
                "timeGPU": data[key][0],
                "timeCPU": data[key][1],
                "avgTimeGPU": data[key][0],
                "avgTimeCPU": data[key][1],
                "color": colorHash.hex(key)
            });
        }
    }

    // remove all with very little contribution
    var minTime = 1000;
    values = values.filter(function (element) {
        return element.timeGPU > minTime || element.timeCPU > minTime ||
            element.avgTimeGPU > minTime || element.avgTimeCPU > minTime;
    });

    // update average values
    var maxValue = 1e9 / 30;
    values.forEach(function (element) {
        var alpha = 0.95;
        element.avgTimeGPU = element.avgTimeGPU * alpha + element.timeGPU * (1 - alpha);
        element.avgTimeCPU = element.avgTimeCPU * alpha + element.timeCPU * (1 - alpha);
        // maxValue = Math.max(maxValue, element.avgTimeGPU);
        // maxValue = Math.max(maxValue, element.avgTimeCPU);
    });

    // sort by average
    values.sort(function (a, b) {
        return (b.avgTimeGPU + b.avgTimeCPU) - (a.avgTimeGPU + a.avgTimeCPU);
    });

    $("#container").html('');
    var maxEntries = Math.min(10, values.length);
    var maxWidth = $("#container").outerWidth();

    $("#container").append(`<div class='label' align='right'><strong>FPS: ${frameRate.toFixed(2)}</strong></div>`);

    for (var i = 0; i < maxEntries; ++i) {
        var widthGPU = maxWidth * values[i].avgTimeGPU / maxValue;
        var widthCPU = maxWidth * values[i].avgTimeCPU / maxValue;
        $("#container").append("<div class='item'></div>");
        $(".item").last().append("<div class='bar gpu' style='background-color:" + values[i].color + "; width:" + widthGPU + "px'><div class='label'>gpu: " + (values[i].avgTimeGPU * 0.000001).toFixed(1) + " ms</div></div>");
        $(".item").last().append("<div class='bar cpu' style='background-color:" + values[i].color + "; width:" + widthCPU + "px'><div class='label'>cpu: " + (values[i].avgTimeCPU * 0.000001).toFixed(1) + " ms</div></div>");
        $(".item").last().append("<div class='label'>" + values[i].name + "</div>");
    }
}

function set_data_test() {
    var newValues = {};
    for (var i = 0; i < 12; ++i) {
        newValues["foo" + i] = [1000000 * Math.random() + i * 100000, 100000 * Math.random() + i * 100000];
    }
    set_data(JSON.stringify(newValues));
}

$(document).ready(function () {
    // set_data_test();
    // window.setInterval(function(){
    //   set_data_test();
    // }, 16);
});