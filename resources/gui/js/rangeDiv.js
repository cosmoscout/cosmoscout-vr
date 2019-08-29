let minWidth = 70;
let offset = 10;
let shorten = 5;

function drawDiv() {
  
    var leftCustomTime = document.getElementsByClassName("leftTime")[0];
    var leftRect = leftCustomTime.getBoundingClientRect();
    var rightCustomTime = document.getElementsByClassName("rightTime")[0];
    var rightRect = rightCustomTime.getBoundingClientRect();
    
    var divElement = document.getElementById("snippet");
    divElement.style.position = "absolute";
    divElement.style.left = leftRect.right + 'px';
    divElement.style.top = (leftRect.top+offset) + 'px';

    let height = leftRect.bottom - leftRect.top - shorten;
    let width = rightRect.right - leftRect.left;

    if(width < minWidth) {
        width = minWidth;
        var xValue = -(leftRect.left + minWidth - rightRect.right) / 2;
        divElement.style.transform = " translate(" + xValue + "px, 0px)";
    } else {
        divElement.style.transform = " translate(0px, 0px)";
    }

    divElement.style.height = height +'px';
    divElement.style.width = width + 'px';
  }

drawDivCallback = drawDiv;