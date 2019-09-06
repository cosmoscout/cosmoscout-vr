let minWidth = 70;
let offset = 5;
let shorten = 5;
let borderWidth = 3;

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
    var width = rightRect.right - leftRect.left;

    var xValue = 0;
    if(width < minWidth) {
        width = minWidth;
        xValue = -(leftRect.left + minWidth - rightRect.right) / 2;
        divElement.style.transform = " translate(" + xValue + "px, 0px)";
    } else {
        divElement.style.transform = " translate(0px, 0px)";
    }

    divElement.style.height = height +'px';
    divElement.style.width = width + 'px';

    divElement = document.getElementById("leftSnippet");
    divElement.style.top = (leftRect.top+offset+height) + 'px';
    width = leftRect.right+xValue+borderWidth;
    width = width < 0 ? 0 : width;
    divElement.style.width = width + 'px';
    var body = document.getElementsByTagName("body")[0];
    var bodyRect = body.getBoundingClientRect();

    divElement = document.getElementById("rightSnippet");
    divElement.style.top = (leftRect.top+offset+height) + 'px';
    width = bodyRect.right-rightRect.right+xValue+1;
    width = width < 0 ? 0 : width;
    divElement.style.width = width + 'px';
  }

drawDivCallback = drawDiv;