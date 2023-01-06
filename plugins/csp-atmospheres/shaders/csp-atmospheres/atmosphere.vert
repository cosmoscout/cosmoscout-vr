#version 330

// uniforms
uniform mat4 uMatInvMV;
uniform mat4 uMatInvP;

// outputs
out VaryingStruct {
  vec3 vRayDir;
  vec3 vRayOrigin;
  vec2 vTexcoords;
}
vsOut;

void main() {
  // get camera position in model space
  vsOut.vRayOrigin = uMatInvMV[3].xyz;

  // get ray direction model space
  mat4 matInvMV = uMatInvMV;
  matInvMV[3]   = vec4(0, 0, 0, 1);
  vec2 position = vec2(gl_VertexID & 2, (gl_VertexID << 1) & 2) * 2.0 - 1.0;
  vsOut.vRayDir = (matInvMV * uMatInvP * vec4(position, 0, 1)).xyz;

  // for lookups in the depth and color buffers
  vsOut.vTexcoords = position * 0.5 + 0.5;

  // no tranformation here since we draw a full screen quad
  gl_Position = vec4(position, 0, 1);
}