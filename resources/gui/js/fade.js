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