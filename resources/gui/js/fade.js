var locked = false;

function toggleLock() {
    document.getElementById('divContainer').classList.toggle('locked');
    document.getElementById('btnLock').classList.toggle('locked');
    locked = !locked;
    if(locked) {
         document.getElementById("btnLock").innerHTML = '<i class="material-icons">lock</i>';
    }
    else {
        document.getElementById("btnLock").innerHTML = '<i class="material-icons">lock_open</i>';
    }
}


function mouseEnterTimeline (){
    if(!locked) {
        document.getElementById('divContainer').classList.add('visible');
    }
}

function mouseLeaveTimenavigation () {
    if(!locked) {
        document.getElementById('divContainer').classList.remove('visible');
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
    document.getElementById("increaseControl").classList.add('mouseOver');
    document.getElementById("decreaseControl").classList.add('mouseOver');
}

function leaveTimeButtons() {
    document.getElementById("increaseControl").classList.remove('mouseOver');
    document.getElementById("decreaseControl").classList.remove('mouseOver');
}

document.getElementById("btnLock").onclick = toggleLock;

document.getElementById("visualization").onmouseenter = mouseEnterTimeline;
document.getElementById("divContainer").onmouseleave = mouseLeaveTimenavigation;

document.getElementById("timeControl").onmouseenter = mouseEnterTimeControl;
document.getElementById("timeControl").onmouseleave = mouseLeaveTimeControl;

document.getElementById("increaseControl").onmouseenter = enterTimeButtons;
document.getElementById("increaseControl").onmouseleave = leaveTimeButtons;

document.getElementById("decreaseControl").onmouseenter = enterTimeButtons;
document.getElementById("decreaseControl").onmouseleave = leaveTimeButtons;