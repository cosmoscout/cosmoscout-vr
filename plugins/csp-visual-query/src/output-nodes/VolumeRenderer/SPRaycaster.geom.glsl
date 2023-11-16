#version 330 core

layout(points) in;
layout(triangle_strip, max_vertices = 3) out;

uniform mat4 uMatInvMV;
uniform mat4 uMatInvP;

out vec3 rayDirection;

void main() {
  mat4 matInvMV = uMatInvMV;
  matInvMV[3]   = vec4(0, 0, 0, 1);

  {
    gl_Position = vec4(-1.0, -1.0, 0.5, 1.0);
    vec2 position = vec2(-1.0, -1.0);
    rayDirection  = (matInvMV * uMatInvP * vec4(position, 0, 1)).xyz;
    EmitVertex();
  }

  {
    gl_Position = vec4(3.0, -1.0, 0.5, 1.0);
    vec2 position = vec2(3.0, -1.0);
    rayDirection  = (matInvMV * uMatInvP * vec4(position, 0, 1)).xyz;
    EmitVertex();
  }

  {
    gl_Position = vec4(-1.0, 3.0, 0.5, 1.0);
    vec2 position = vec2(-1.0, 3.0);
    rayDirection  = (matInvMV * uMatInvP * vec4(position, 0, 1)).xyz;
    EmitVertex();
  }

  EndPrimitive();
}