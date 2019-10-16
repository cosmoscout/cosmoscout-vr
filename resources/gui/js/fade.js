var overviewVisible = false;

// Toggles if the timeline is locked or is able to fade in/out
function toggleLock() {
    overviewVisible = !overviewVisible;
    document.getElementById('divContainer').classList.toggle('visible');
    if (overviewVisible) {
        document.getElementById("btnExpand").innerHTML = '<i class="material-icons">expand_less</i>';
    }
    else {
        document.getElementById("btnExpand").innerHTML = '<i class="material-icons">expand_more</i>';
    }
}

function mouseEnterTimeControl() {
    document.getElementById("increaseControl").classList.add('mouseNear');
    document.getElementById("decreaseControl").classList.add('mouseNear');
}

function mouseLeaveTimeControl() {
    document.getElementById("increaseControl").classList.remove('mouseNear');
    document.getElementById("decreaseControl").classList.remove('mouseNear');
}

function enterTimeButtons() {
    document.getElementById("increaseControl").classList.add('mouseNear');
    document.getElementById("decreaseControl").classList.add('mouseNear');
}

function leaveTimeButtons() {
    document.getElementById("increaseControl").classList.remove('mouseNear');
    document.getElementById("decreaseControl").classList.remove('mouseNear');
}

document.getElementById("btnExpand").onclick = toggleLock;

document.getElementById("timeControl").onmouseenter = mouseEnterTimeControl;
document.getElementById("timeControl").onmouseleave = mouseLeaveTimeControl;

document.getElementById("increaseControl").onmouseenter = enterTimeButtons;
document.getElementById("increaseControl").onmouseleave = leaveTimeButtons;

document.getElementById("decreaseControl").onmouseenter = enterTimeButtons;
document.getElementById("decreaseControl").onmouseleave = leaveTimeButtons;