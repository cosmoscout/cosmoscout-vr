import Agent from './agent.js'
import Animator from './animator.js'
import Queue from './queue.js'
import Balloon from './balloon.js'
import { load, ready, soundsReady } from './load.js'

const clippy = {
    Agent,
    Animator,
    Queue,
    Balloon,
    load,
    ready,
    soundsReady
}

export default clippy

if (typeof window !== 'undefined') {
    window.clippy = clippy
}


