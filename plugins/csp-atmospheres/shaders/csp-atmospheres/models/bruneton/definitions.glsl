////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: 2017 Eric Bruneton
// SPDX-License-Identifier: BSD-3-Clause

// This file has been directly copied from here:
// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/definitions.glsl
// The only differences should be related to formatting. The documentation below can also be read
// online at:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/definitions.glsl.html

/*<h2>atmosphere/definitions.glsl</h2>

<p>This GLSL file defines the physical types and constants which are used in the
main <a href="functions.glsl.html">functions</a> of our atmosphere model, in
such a way that they can be compiled by a GLSL compiler (a
<a href="reference/definitions.h.html">C++ equivalent</a> of this file
provides the same types and constants in C++, to allow the same functions to be
compiled by a C++ compiler - see the <a href="../index.html">Introduction</a>).

<h3>Physical quantities</h3>

<p>The physical quantities we need for our atmosphere model are
<a href="https://en.wikipedia.org/wiki/Radiometry">radiometric</a> and
<a href="https://en.wikipedia.org/wiki/Photometry_(optics)">photometric</a>
quantities. In GLSL we can't define custom numeric types to enforce the
homogeneity of expressions at compile time, so we define all the physical
quantities as <code>float</code>, with preprocessor macros (there is no
<code>typedef</code> in GLSL).

<p>We start with six base quantities: length, wavelength, angle, solid angle,
power and luminous power (wavelength is also a length, but we distinguish the
two for increased clarity).
*/

/*
<p>We  also need vectors of physical quantities, mostly to represent functions
depending on the wavelength. In this case the vector elements correspond to
values of a function at some predefined wavelengths. Again, in GLSL we can't
define custom vector types to enforce the homogeneity of expressions at compile
time, so we define these vector types as <code>vec3</code>, with preprocessor
macros. The full definitions are given in the
<a href="reference/definitions.h.html">C++ equivalent</a> of this file).
*/

/*
<p>Finally, we also need precomputed textures containing physical quantities in
each texel. Since we can't define custom sampler types to enforce the
homogeneity of expressions at compile time in GLSL, we define these texture
types as <code>sampler2D</code> and <code>sampler3D</code>, with preprocessor
macros. The full definitions are given in the
<a href="reference/definitions.h.html">C++ equivalent</a> of this file).
*/

/*
<h3>Physical units</h3>

<p>We can then define the units for our six base physical quantities:
meter (m), nanometer (nm), radian (rad), steradian (sr), watt (watt) and lumen
(lm):
*/

/*
<p>From which we can derive the units for some derived physical quantities,
as well as some derived units (kilometer km, kilocandela kcd, degree deg):
*/

const float PI = 3.14159265358979323846;

/*
<h3>Atmosphere parameters</h3>

<p>Using the above types, we can now define the parameters of our atmosphere
model.
*/

// This texture contains all the phase functions for the scattering components of the atmosphere.
// The u coordinate maps to the scattering angle. Forward scattering is on the left (u == 0, theta
// == 0) and back scattering on the right (u == 1, theta == 180).
// The v coordinate maps to the various components. The phase function of the first scattering
// component is stored in the top row of pixels, the second in the next and so on.
uniform sampler2D phase_texture;

// This texture contains all the density functions for the components of the atmosphere.
// The u coordinate maps to the altitude. Atmosphere's bottom is on the left (u == 0) and the top
// is the right (u == 1). The v coordinate maps to the various components. The density function of
// the first component is stored in the top row of pixels, the second in the next and so on.
uniform sampler2D density_texture;

struct ScatteringComponent {
  float phaseTextureV;
  float densityTextureV;
  vec3  extinction;
  vec3  scattering;
};

struct AbsorbingComponent {
  float densityTextureV;
  vec3  extinction;
};

struct AtmosphereComponents {
  ScatteringComponent molecules;
  ScatteringComponent aerosols;
  AbsorbingComponent  ozone;
};
