var locked = false;

function toggleLock() {
    document.getElementById('divContainer').classList.toggle('locked');
    document.getElementById('btnLock').classList.toggle('locked');
    locked = !locked;
}


function mouseEnterCallback (evt){
    if(!locked) {
        document.getElementById('divContainer').classList.add('visible');
    }
}

function mouseLeaveCallback (evt) {
    if(!locked) {
        document.getElementById('divContainer').classList.remove('visible');
    }
}

document.getElementById("btnLock").onclick = toggleLock;

document.getElementById("visualization").onmouseenter = mouseEnterCallback;
document.getElementById("divContainer").onmouseleave = mouseLeaveCallback;