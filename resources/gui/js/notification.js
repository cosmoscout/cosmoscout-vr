function print_notification(title, content, icon, flyto) {
    $("#notifications").prepend(
        "<div class='notification row'>" +
        "<div class='col-2'>" +
        "<i class='material-icons'>" + icon + "</i>" +
        "</div>" +
        "<div class='col-10'>" +
        "<div class='notification-title'>" + title + "</div>" +
        "<div>" + content + "</div>" +
        "</div>" +
        "</div>"
    );

    if (flyto) {
        $(".notification").first().addClass("clickable");
        $(".notification").first().click(function () {
            window.call_native("fly_to", flyto);
        });
    }

    $(".notification").first().css("transform");
    $(".notification").first().addClass("show");
    $(".notification").first().delay(8000).queue(function (next) {
        $(this).addClass("fadeout");
        next();
    });

    $(".notification")
        .not(":nth-child(1)")
        .not(":nth-child(2)")
        .not(":nth-child(3)")
        .not(":nth-child(4)")
        .not(":nth-child(5)")
        .remove();
}

