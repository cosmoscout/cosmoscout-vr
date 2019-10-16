var picker = new CP(document.querySelector('input[type="colorPicker"]'));
picker.on("change", function (color) {
    this.source.value = '#' + color;
});

picker.on("change", function (color) {
    var colorField = document.getElementById("eventColor");
    colorField.style.background = '#' + color;
});
