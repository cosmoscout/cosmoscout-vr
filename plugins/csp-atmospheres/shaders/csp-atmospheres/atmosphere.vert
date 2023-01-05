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
  mat4 testInvMV = uMatInvMV;
  testInvMV[3]   = vec4(0, 0, 0, 1);

  mat4 testInvMVP = testInvMV * uMatInvP;

  // get camera position in model space
  vsOut.vRayOrigin = uMatInvMV[3].xyz;

  // get ray direction model space
  vec2 vPosition = vec2(gl_VertexID & 2, (gl_VertexID << 1) & 2) * 2.0 - 1.0;
  vsOut.vRayDir  = (testInvMVP * vec4(vPosition, 0, 1)).xyz;

  // for lookups in the depth and color buffers
  vsOut.vTexcoords = vPosition * 0.5 + 0.5;

  // no tranformation here since we draw a full screen quad
  gl_Position = vec4(vPosition, 0, 1);
}