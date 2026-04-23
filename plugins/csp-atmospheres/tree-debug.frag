#version 440

in vec3 pos;
out vec4 color;

void main() {
    color = vec4(1, 1, 1, 1);
    gl_FragDepth = length(inPos);
}