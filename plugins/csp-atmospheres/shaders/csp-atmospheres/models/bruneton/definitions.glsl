////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

const float PI = 3.14159265358979323846;

// This texture contains all the phase functions for the scattering components of the atmosphere.
// The u coordinate maps to the scattering angle. Forward scattering is on the left (u == 0, theta
// == 0) and back scattering on the right (u == 1, theta == 180).
// The v coordinate maps to the various components. The phase function of the first scattering
// component is stored in the top row of pixels, the second in the next and so on. So usually the
// phase function for molecules is in the top row and the phase function for aerosols is in the
// bottom row.
uniform sampler2D uPhaseTexture;

// This texture contains all the density functions for the components of the atmosphere. The density
// function maps a relative density value in [0..1] to each altitude. The u coordinate corresponds
// to the altitude; atmosphere's bottom is on the left (u == 0) and the top is the right (u == 1).
// The v coordinate maps to the various components. The density function of the first component is
// stored in the top row of pixels, the second in the next and so on. So usually the molecules are
// in the first row, aerosols in the second row, and ozone is in the last row.
// The density_texture is only sampled during preprocessing. It is not used at runtime in the final
// fragment shader.
uniform sampler2D uDensityTexture;

// Scattering components can absorb and scatter light. Air molecules and aerosols in Earth's
// atmosphere are both modelled using scattering components.
struct ScatteringComponent {

  // The vertical texture coordinate of the component's the phse function in phase_texture.
  float phaseTextureV;

  // The vertical texture coordinate of the component's the density function in density_texture.
  float densityTextureV;

  // The extinction coefficient beta_ext in m^-1.
  vec3 extinction;

  // The extinction coefficient beta_sca in m^-1.
  vec3 scattering;
};

// Absorbing components can absorb light but do not scatter light. Ozone in Earth's atmosphere is
// modelled using an absorbing component.
struct AbsorbingComponent {

  // The vertical texture coordinate of the component's the density function in density_texture.
  float densityTextureV;

  // The extinction coefficient beta_ext in m^-1.
  vec3 extinction;
};

// An atmosphere consists of two scattering components and an absorbing component. The scattering
// components can absorb light as well, but the absorbing component can not scatter light.
struct AtmosphereComponents {
  ScatteringComponent molecules;
  ScatteringComponent aerosols;
  AbsorbingComponent  ozone;
};
