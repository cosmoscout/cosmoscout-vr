#version 440

layout (location = 0) in vec3 inPos;
out vec3 pos;

// uniform float planetRadius;

void main() {
    pos = inPos;
}