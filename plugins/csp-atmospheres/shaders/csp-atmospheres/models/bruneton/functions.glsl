////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: 2017 Eric Bruneton
// SPDX-FileCopyrightText: 2008 INRIA
// SPDX-License-Identifier: BSD-3-Clause

// This file has been directly copied from here:
// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/functions.glsl
// The documentation below can also be read online at:
// https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html

// The only difference with respect to the original implementation is the removal of the "normal"
// parameter from GetSunAndSkyIrradiance() at the very end of the file. In the original
// implementation, the method used to premultiply the irradiance with the dot product between light
// direction and surface normal. As this factor is already included in the BRDFs used in CosmoCout
// VR, we have removed this.

/*<h2>atmosphere/functions.glsl</h2>

<p>This GLSL file contains the core functions that implement our atmosphere
model. It provides functions to compute the transmittance, the single scattering
and the second and higher orders of scattering, the ground irradiance, as well
as functions to store these in textures and to read them back. It uses physical
types and constants which are provided in two versions: a
<a href="definitions.glsl.html">GLSL version</a> and a
<a href="reference/definitions.h.html">C++ version</a>. This allows this file to
be compiled either with a GLSL compiler or with a C++ compiler (see the
<a href="../index.html">Introduction</a>).

<p>The functions provided in this file are organized as follows:
<ul>
<li><a href="#transmittance">Transmittance</a>
<ul>
<li><a href="#transmittance_computation">Computation</a></li>
<li><a href="#transmittance_precomputation">Precomputation</a></li>
<li><a href="#transmittance_lookup">Lookup</a></li>
</ul>
</li>
<li><a href="#single_scattering">Single scattering</a>
<ul>
<li><a href="#single_scattering_computation">Computation</a></li>
<li><a href="#single_scattering_precomputation">Precomputation</a></li>
<li><a href="#single_scattering_lookup">Lookup</a></li>
</ul>
</li>
<li><a href="#multiple_scattering">Multiple scattering</a>
<ul>
<li><a href="#multiple_scattering_computation">Computation</a></li>
<li><a href="#multiple_scattering_precomputation">Precomputation</a></li>
<li><a href="#multiple_scattering_lookup">Lookup</a></li>
</ul>
</li>
<li><a href="#irradiance">Ground irradiance</a>
<ul>
<li><a href="#irradiance_computation">Computation</a></li>
<li><a href="#irradiance_precomputation">Precomputation</a></li>
<li><a href="#irradiance_lookup">Lookup</a></li>
</ul>
</li>
<li><a href="#rendering">Rendering</a>
<ul>
<li><a href="#rendering_sky">Sky</a></li>
<li><a href="#rendering_aerial_perspective">Aerial perspective</a></li>
<li><a href="#rendering_ground">Ground</a></li>
</ul>
</li>
</ul>

<p>They use the following utility functions to avoid NaNs due to floating point
values slightly outside their theoretical bounds:
*/

float ClampCosine(float mu) {
  return clamp(mu, -1.0, 1.0);
}

float ClampDistance(float d) {
  return max(d, 0.0);
}

float ClampRadius(float r) {
  return clamp(r, BOTTOM_RADIUS, TOP_RADIUS);
}

float SafeSqrt(float a) {
  return sqrt(max(a, 0.0));
}

/*
<h3 id="transmittance">Transmittance</h3>

<p>As the light travels from a point $\bp$ to a point $\bq$ in the atmosphere,
it is partially absorbed and scattered out of its initial direction because of
the air molecules and the aerosol particles. Thus, the light arriving at $\bq$
is only a fraction of the light from $\bp$, and this fraction, which depends on
wavelength, is called the
<a href="https://en.wikipedia.org/wiki/Transmittance">transmittance</a>. The
following sections describe how we compute it, how we store it in a precomputed
texture, and how we read it back.

<h4 id="transmittance_computation">Computation</h4>

<p>For 3 aligned points $\bp$, $\bq$ and $\br$ inside the atmosphere, in this
order, the transmittance between $\bp$ and $\br$ is the product of the
transmittance between $\bp$ and $\bq$ and between $\bq$ and $\br$. In
particular, the transmittance between $\bp$ and $\bq$ is the transmittance
between $\bp$ and the nearest intersection $\bi$ of the half-line $[\bp,\bq)$
with the top or bottom atmosphere boundary, divided by the transmittance between
$\bq$ and $\bi$ (or 0 if the segment $[\bp,\bq]$ intersects the ground):

<svg width="340px" height="195px">
  <style type="text/css"><![CDATA[
    circle { fill: #000000; stroke: none; }
    path { fill: none; stroke: #000000; }
    text { font-size: 16px; font-style: normal; font-family: Sans; }
    .vector { font-weight: bold; }
  ]]></style>
  <path d="m 0,26 a 600,600 0 0 1 340,0"/>
  <path d="m 0,110 a 520,520 0 0 1 340,0"/>
  <path d="m 170,190 0,-30"/>
  <path d="m 170,140 0,-130"/>
  <path d="m 170,50 165,-33"/>
  <path d="m 155,150 10,-10 10,10 10,-10"/>
  <path d="m 155,160 10,-10 10,10 10,-10"/>
  <path d="m 95,50 30,0"/>
  <path d="m 95,190 30,0"/>
  <path d="m 105,50 0,140" style="stroke-dasharray:8,2;"/>
  <path d="m 100,55 5,-5 5,5"/>
  <path d="m 100,185 5,5 5,-5"/>
  <path d="m 170,25 a 25,25 0 0 1 25,20" style="stroke-dasharray:4,2;"/>
  <path d="m 170,190 70,0"/>
  <path d="m 235,185 5,5 -5,5"/>
  <path d="m 165,125 5,-5 5,5"/>
  <circle cx="170" cy="190" r="2.5"/>
  <circle cx="170" cy="50" r="2.5"/>
  <circle cx="320" cy="20" r="2.5"/>
  <circle cx="270" cy="30" r="2.5"/>
  <text x="155" y="45" class="vector">p</text>
  <text x="265" y="45" class="vector">q</text>
  <text x="320" y="15" class="vector">i</text>
  <text x="175" y="185" class="vector">o</text>
  <text x="90" y="125">r</text>
  <text x="185" y="25">μ=cos(θ)</text>
  <text x="240" y="185">x</text>
  <text x="155" y="120">z</text>
</svg>

<p>Also, the transmittance between $\bp$ and $\bq$ and between $\bq$ and $\bp$
are the same. Thus, to compute the transmittance between arbitrary points, it
is sufficient to know the transmittance between a point $\bp$ in the atmosphere,
and points $\bi$ on the top atmosphere boundary. This transmittance depends on
only two parameters, which can be taken as the radius $r=\Vert\bo\bp\Vert$ and
the cosine of the "view zenith angle",
$\mu=\bo\bp\cdot\bp\bi/\Vert\bo\bp\Vert\Vert\bp\bi\Vert$. To compute it, we
first need to compute the length $\Vert\bp\bi\Vert$, and we need to know when
the segment $[\bp,\bi]$ intersects the ground.

<h5>Distance to the top atmosphere boundary</h5>

<p>A point at distance $d$ from $\bp$ along $[\bp,\bi)$ has coordinates
$[d\sqrt{1-\mu^2}, r+d\mu]^\top$, whose squared norm is $d^2+2r\mu d+r^2$.
Thus, by definition of $\bi$, we have
$\Vert\bp\bi\Vert^2+2r\mu\Vert\bp\bi\Vert+r^2=r_{\mathrm{top}}^2$,
from which we deduce the length $\Vert\bp\bi\Vert$:
*/

float DistanceToTopAtmosphereBoundary(float r, float mu) {
  assert(r <= TOP_RADIUS);
  assert(mu >= -1.0 && mu <= 1.0);
  float discriminant = r * r * (mu * mu - 1.0) + TOP_RADIUS * TOP_RADIUS;
  return ClampDistance(-r * mu + SafeSqrt(discriminant));
}

/*
<p>We will also need, in the other sections, the distance to the bottom
atmosphere boundary, which can be computed in a similar way (this code assumes
that $[\bp,\bi)$ intersects the ground):
*/

float DistanceToBottomAtmosphereBoundary(float r, float mu) {
  assert(r >= BOTTOM_RADIUS);
  assert(mu >= -1.0 && mu <= 1.0);
  float discriminant = r * r * (mu * mu - 1.0) + BOTTOM_RADIUS * BOTTOM_RADIUS;
  return ClampDistance(-r * mu - SafeSqrt(discriminant));
}

/*
<h5>Intersections with the ground</h5>

<p>The segment $[\bp,\bi]$ intersects the ground when
$d^2+2r\mu d+r^2=r_{\mathrm{bottom}}^2$ has a solution with $d \ge 0$. This
requires the discriminant $r^2(\mu^2-1)+r_{\mathrm{bottom}}^2$ to be positive,
from which we deduce the following function:
*/

bool RayIntersectsGround(float r, float mu) {
  assert(r >= BOTTOM_RADIUS);
  assert(mu >= -1.0 && mu <= 1.0);
  return mu < 0.0 && r * r * (mu * mu - 1.0) + BOTTOM_RADIUS * BOTTOM_RADIUS >= 0.0;
}

/*
<h5>Transmittance to the top atmosphere boundary</h5>

<p>We can now compute the transmittance between $\bp$ and $\bi$. From its
definition and the
<a href="https://en.wikipedia.org/wiki/Beer-Lambert_law">Beer-Lambert law</a>,
this involves the integral of the float density of air molecules along the
segment $[\bp,\bi]$, as well as the integral of the float density of aerosols
and the integral of the float density of air molecules that absorb light
(e.g. ozone) - along the same segment. These 3 integrals have the same form and,
when the segment $[\bp,\bi]$ does not intersect the ground, they can be computed
numerically with the help of the following auxilliary function (using the <a
href="https://en.wikipedia.org/wiki/Trapezoidal_rule">trapezoidal rule</a>):
*/

float GetDensity(IN(float) densityTextureV, float altitude) {
  float u = clamp(altitude / (TOP_RADIUS - BOTTOM_RADIUS), 0.0, 1.0);
  return texture(density_texture, vec2(u, densityTextureV)).r;
}

float ComputeOpticalLengthToTopAtmosphereBoundary(IN(float) densityTextureV, float r, float mu) {
  assert(r >= BOTTOM_RADIUS && r <= TOP_RADIUS);
  assert(mu >= -1.0 && mu <= 1.0);

  // The integration step, i.e. the length of each integration interval.
  float dx = DistanceToTopAtmosphereBoundary(r, mu) / float(SAMPLE_COUNT_OPTICAL_DEPTH);
  // Integration loop.
  float result = 0.0;
  for (int i = 0; i <= SAMPLE_COUNT_OPTICAL_DEPTH; ++i) {
    float d_i = float(i) * dx;
    // Distance between the current sample point and the planet center.
    float r_i = sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r);
    // float density at the current sample point (divided by the float density
    // at the bottom of the atmosphere, yielding a dimensionless float).
    float y_i = GetDensity(densityTextureV, r_i - BOTTOM_RADIUS);
    // Sample weight (from the trapezoidal rule).
    float weight_i = i == 0 || i == SAMPLE_COUNT_OPTICAL_DEPTH ? 0.5 : 1.0;
    result += y_i * weight_i * dx;
  }
  return result;
}

/*
<p>With this function the transmittance between $\bp$ and $\bi$ is now easy to
compute (we continue to assume that the segment does not intersect the ground):
*/

vec3 ComputeTransmittanceToTopAtmosphereBoundary(
    IN(AtmosphereComponents) atmosphere, float r, float mu) {
  assert(r >= BOTTOM_RADIUS && r <= TOP_RADIUS);
  assert(mu >= -1.0 && mu <= 1.0);
  return exp(-(atmosphere.molecules.extinction * ComputeOpticalLengthToTopAtmosphereBoundary(
                                                     atmosphere.molecules.densityTextureV, r, mu) +
               atmosphere.aerosols.extinction * ComputeOpticalLengthToTopAtmosphereBoundary(
                                                    atmosphere.aerosols.densityTextureV, r, mu) +
               atmosphere.ozone.extinction * ComputeOpticalLengthToTopAtmosphereBoundary(
                                                 atmosphere.ozone.densityTextureV, r, mu)));
}

/*
<h4 id="transmittance_precomputation">Precomputation</h4>

<p>The above function is quite costly to evaluate, and a lot of evaluations are
needed to compute single and multiple scattering. Fortunately this function
depends on only two parameters and is quite smooth, so we can precompute it in a
small 2D texture to optimize its evaluation.

<p>For this we need a mapping between the function parameters $(r,\mu)$ and the
texture coordinates $(u,v)$, and vice-versa, because these parameters do not
have the same units and range of values. And even if it was the case, storing a
function $f$ from the $[0,1]$ interval in a texture of size $n$ would sample the
function at $0.5/n$, $1.5/n$, ... $(n-0.5)/n$, because texture samples are at
the center of texels. Therefore, this texture would only give us extrapolated
function values at the domain boundaries ($0$ and $1$). To avoid this we need
to store $f(0)$ at the center of texel 0 and $f(1)$ at the center of texel
$n-1$. This can be done with the following mapping from values $x$ in $[0,1]$ to
texture coordinates $u$ in $[0.5/n,1-0.5/n]$ - and its inverse:
*/

float GetTextureCoordFromUnitRange(float x, int texture_size) {
  return 0.5 / float(texture_size) + x * (1.0 - 1.0 / float(texture_size));
}

float GetUnitRangeFromTextureCoord(float u, int texture_size) {
  return (u - 0.5 / float(texture_size)) / (1.0 - 1.0 / float(texture_size));
}

/*
<p>Using these functions, we can now define a mapping between $(r,\mu)$ and the
texture coordinates $(u,v)$, and its inverse, which avoid any extrapolation
during texture lookups. In the <a href=
"http://evasion.inrialpes.fr/~Eric.Bruneton/PrecomputedAtmosphericScattering2.zip"
>original implementation</a> this mapping was using some ad-hoc constants chosen
for the Earth atmosphere case. Here we use a generic mapping, working for any
atmosphere, but still providing an increased sampling rate near the horizon.
Our improved mapping is based on the parameterization described in our
<a href="https://hal.inria.fr/inria-00288758/en">paper</a> for the 4D textures:
we use the same mapping for $r$, and a slightly improved mapping for $\mu$
(considering only the case where the view ray does not intersect the ground).
More precisely, we map $\mu$ to a value $x_{\mu}$ between 0 and 1 by considering
the distance $d$ to the top atmosphere boundary, compared to its minimum and
maximum values $d_{\mathrm{min}}=r_{\mathrm{top}}-r$ and
$d_{\mathrm{max}}=\rho+H$ (cf. the notations from the
<a href="https://hal.inria.fr/inria-00288758/en">paper</a> and the figure
below):

<svg width="505px" height="195px">
  <style type="text/css"><![CDATA[
    circle { fill: #000000; stroke: none; }
    path { fill: none; stroke: #000000; }
    text { font-size: 16px; font-style: normal; font-family: Sans; }
    .vector { font-weight: bold; }
  ]]></style>
  <path d="m 5,85 a 520,520 0 0 1 372,105"/>
  <path d="m 5,5 a 600,600 0 0 1 490,185"/>
  <path d="m 60,0 0,190"/>
  <path d="m 60,65 180,-35"/>
  <path d="m 55,5 5,-5 5,5"/>
  <path d="m 55,60 5,5 5,-5"/>
  <path d="m 55,70 5,-5 5,5"/>
  <path d="m 60,40 a 25,25 0 0 1 25,20" style="stroke-dasharray:4,2;"/>
  <path d="m 60,65 415,105"/>
  <circle cx="60" cy="65" r="2.5"/>
  <circle cx="240" cy="30" r="2.5"/>
  <circle cx="180" cy="95" r="2.5"/>
  <circle cx="475" cy="170" r="2.5"/>
  <text x="20" y="40">d<tspan style="font-size:10px" dy="2">min</tspan></text>
  <text x="35" y="70" class="vector">p</text>
  <text x="35" y="125">r</text>
  <text x="75" y="40">μ=cos(θ)</text>
  <text x="120" y="75">ρ</text>
  <text x="155" y="60">d</text>
  <text x="315" y="125">H</text>
</svg>

<p>With these definitions, the mapping from $(r,\mu)$ to the texture coordinates
$(u,v)$ can be implemented as follows:
*/

vec2 GetTransmittanceTextureUvFromRMu(float r, float mu) {
  assert(r >= BOTTOM_RADIUS && r <= TOP_RADIUS);
  assert(mu >= -1.0 && mu <= 1.0);
  // Distance to top atmosphere boundary for a horizontal ray at ground level.
  float H = sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the horizon.
  float rho = SafeSqrt(r * r - BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
  // and maximum values over all mu - obtained for (r,1) and (r,mu_horizon).
  float d     = DistanceToTopAtmosphereBoundary(r, mu);
  float d_min = TOP_RADIUS - r;
  float d_max = rho + H;
  float x_mu  = (d - d_min) / (d_max - d_min);
  float x_r   = rho / H;
  return vec2(GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_TEXTURE_WIDTH),
      GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_TEXTURE_HEIGHT));
}

/*
<p>and the inverse mapping follows immediately:
*/

void GetRMuFromTransmittanceTextureUv(IN(vec2) uv, OUT(float) r, OUT(float) mu) {
  assert(uv.x >= 0.0 && uv.x <= 1.0);
  assert(uv.y >= 0.0 && uv.y <= 1.0);
  float x_mu = GetUnitRangeFromTextureCoord(uv.x, TRANSMITTANCE_TEXTURE_WIDTH);
  float x_r  = GetUnitRangeFromTextureCoord(uv.y, TRANSMITTANCE_TEXTURE_HEIGHT);
  // Distance to top atmosphere boundary for a horizontal ray at ground level.
  float H = sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the horizon, from which we can compute r:
  float rho = H * x_r;
  r         = sqrt(rho * rho + BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
  // and maximum values over all mu - obtained for (r,1) and (r,mu_horizon) -
  // from which we can recover mu:
  float d_min = TOP_RADIUS - r;
  float d_max = rho + H;
  float d     = d_min + x_mu * (d_max - d_min);
  mu          = d == 0.0 ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * r * d);
  mu          = ClampCosine(mu);
}

/*
<p>It is now easy to define a fragment shader function to precompute a texel of
the transmittance texture:
*/

vec3 ComputeTransmittanceToTopAtmosphereBoundaryTexture(
    IN(AtmosphereComponents) atmosphere, IN(vec2) frag_coord) {
  const vec2 TRANSMITTANCE_TEXTURE_SIZE =
      vec2(TRANSMITTANCE_TEXTURE_WIDTH, TRANSMITTANCE_TEXTURE_HEIGHT);
  float r;
  float mu;
  GetRMuFromTransmittanceTextureUv(frag_coord / TRANSMITTANCE_TEXTURE_SIZE, r, mu);
  return ComputeTransmittanceToTopAtmosphereBoundary(atmosphere, r, mu);
}

/*
<h4 id="transmittance_lookup">Lookup</h4>

<p>With the help of the above precomputed texture, we can now get the
transmittance between a point and the top atmosphere boundary with a single
texture lookup (assuming there is no intersection with the ground):
*/

vec3 GetTransmittanceToTopAtmosphereBoundary(
    IN(sampler2D) transmittance_texture, float r, float mu) {
  assert(r >= BOTTOM_RADIUS && r <= TOP_RADIUS);
  vec2 uv = GetTransmittanceTextureUvFromRMu(r, mu);
  return vec3(texture(transmittance_texture, uv));
}

/*
<p>Also, with $r_d=\Vert\bo\bq\Vert=\sqrt{d^2+2r\mu d+r^2}$ and $\mu_d=
\bo\bq\cdot\bp\bi/\Vert\bo\bq\Vert\Vert\bp\bi\Vert=(r\mu+d)/r_d$ the values of
$r$ and $\mu$ at $\bq$, we can get the transmittance between two arbitrary
points $\bp$ and $\bq$ inside the atmosphere with only two texture lookups
(recall that the transmittance between $\bp$ and $\bq$ is the transmittance
between $\bp$ and the top atmosphere boundary, divided by the transmittance
between $\bq$ and the top atmosphere boundary, or the reverse - we continue to
assume that the segment between the two points does not intersect the ground):
*/

vec3 GetTransmittance(IN(sampler2D) transmittance_texture, float r, float mu, float d,
    bool ray_r_mu_intersects_ground) {
  assert(r >= BOTTOM_RADIUS && r <= TOP_RADIUS);
  assert(mu >= -1.0 && mu <= 1.0);
  assert(d >= 0.0);

  float r_d  = ClampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
  float mu_d = ClampCosine((r * mu + d) / r_d);

  if (ray_r_mu_intersects_ground) {
    return min(GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r_d, -mu_d) /
                   GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r, -mu),
        vec3(1.0));
  } else {
    return min(GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r, mu) /
                   GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r_d, mu_d),
        vec3(1.0));
  }
}

/*
<p>where <code>ray_r_mu_intersects_ground</code> should be true iif the ray
defined by $r$ and $\mu$ intersects the ground. We don't compute it here with
<code>RayIntersectsGround</code> because the result could be wrong for rays
very close to the horizon, due to the finite precision and rounding errors of
floating point operations. And also because the caller generally has more robust
ways to know whether a ray intersects the ground or not (see below).

<p>Finally, we will also need the transmittance between a point in the
atmosphere and the Sun. The Sun is not a point light source, so this is an
integral of the transmittance over the Sun disc. Here we consider that the
transmittance is constant over this disc, except below the horizon, where the
transmittance is 0. As a consequence, the transmittance to the Sun can be
computed with <code>GetTransmittanceToTopAtmosphereBoundary</code>, times the
fraction of the Sun disc which is above the horizon.

<p>This fraction varies from 0 when the Sun zenith angle $\theta_s$ is larger
than the horizon zenith angle $\theta_h$ plus the Sun angular radius $\alpha_s$,
to 1 when $\theta_s$ is smaller than $\theta_h-\alpha_s$. Equivalently, it
varies from 0 when $\mu_s=\cos\theta_s$ is smaller than
$\cos(\theta_h+\alpha_s)\approx\cos\theta_h-\alpha_s\sin\theta_h$ to 1 when
$\mu_s$ is larger than
$\cos(\theta_h-\alpha_s)\approx\cos\theta_h+\alpha_s\sin\theta_h$. In between,
the visible Sun disc fraction varies approximately like a smoothstep (this can
be verified by plotting the area of <a
href="https://en.wikipedia.org/wiki/Circular_segment">circular segment</a> as a
function of its <a href="https://en.wikipedia.org/wiki/Sagitta_(geometry)"
>sagitta</a>). Therefore, since $\sin\theta_h=r_{\mathrm{bottom}}/r$, we can
approximate the transmittance to the Sun with the following function:
*/

vec3 GetTransmittanceToSun(IN(sampler2D) transmittance_texture, float r, float mu_s) {
  float sin_theta_h = BOTTOM_RADIUS / r;
  float cos_theta_h = -sqrt(max(1.0 - sin_theta_h * sin_theta_h, 0.0));
  return GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r, mu_s) *
         smoothstep(-sin_theta_h * SUN_ANGULAR_RADIUS, sin_theta_h * SUN_ANGULAR_RADIUS,
             mu_s - cos_theta_h);
}

/*
<h3 id="single_scattering">Single scattering</h3>

<p>The single scattered radiance is the light arriving from the Sun at some
point after exactly one scattering event inside the atmosphere (which can be due
to air molecules or aerosol particles; we exclude reflections from the ground,
computed <a href="#irradiance">separately</a>). The following sections describe
how we compute it, how we store it in a precomputed texture, and how we read it
back.

<h4 id="single_scattering_computation">Computation</h4>

<p>Consider the Sun light scattered at a point $\bq$ by air molecules before
arriving at another point $\bp$ (for aerosols, replace "Rayleigh" with "Mie"
below):

<svg height="190px" width="340px">
  <style type="text/css"><![CDATA[
    circle { fill: #000000; stroke: none; }
    path { fill: none; stroke: #000000; }
    text { font-size: 16px; font-style: normal; font-family: Sans; }
    .vector { font-weight: bold; }
  ]]></style>
  <path d="m 0,66 a 600,600 0 0 1 340,0"/>
  <path d="m 0,150 a 520,520 0 0 1 340,0"/>
  <path d="m 170,180 0,-165"/>
  <path d="m 250,180 30,-165"/>
  <path d="m 170,90 -30,-60"/>
  <path d="m 155,70 0,-10 8,6" />
  <path d="m 270,70 -20,-40" style="stroke-width:2;"/>
  <path d="m 170,90 100,-20" style="stroke-width:2;"/>
  <path d="m 270,70 75,-15" />
  <path d="m 170,65 a 25,25 0 0 1 25,20" style="stroke-dasharray:4,2;"/>
  <path d="m 170,30 a 60,60 1 0 0 -26.8,6.3" style="stroke-dasharray:4,2;"/>
  <path d="m 255,40 a 35,35 0 0 1 21,-3.2" style="stroke-dasharray:4,2;"/>
  <path d="m 258,45 a 30,30 0 0 1 41,19" style="stroke-dasharray:4,2;"/>
  <circle cx="170" cy="90" r="2.5"/>
  <circle cx="270" cy="70" r="2.5"/>
  <text x="155" y="105" class="vector">p</text>
  <text x="275" y="85" class="vector">q</text>
  <text x="130" y="70" class="vector">ω<tspan
      dy="2" style="font-size:10px;font-weight:normal;">s</tspan></text>
  <text x="155" y="164">r</text>
  <text x="265" y="165">r<tspan dy="2" style="font-size:10px">d</tspan></text>
  <text x="220" y="95">d</text>
  <text x="190" y="65">μ</text>
  <text x="145" y="25">μ<tspan dy="2" style="font-size:10px">s</tspan></text>
  <text x="290" y="45">ν</text>
  <text x="250" y="20">μ<tspan dy="2" style="font-size:10px">s,d</tspan></text>
</svg>

<p>The radiance arriving at $\bp$ is the product of:
<ul>
<li>the solar irradiance at the top of the atmosphere,</li>
<li>the transmittance between the Sun and $\bq$ (i.e. the fraction of the Sun
light at the top of the atmosphere that reaches $\bq$),</li>
<li>the Rayleigh scattering coefficient at $\bq$ (i.e. the fraction of the
light arriving at $\bq$ which is scattered, in any direction),</li>
<li>the Rayleigh phase function (i.e. the fraction of the scattered light at
$\bq$ which is actually scattered towards $\bp$),</li>
<li>the transmittance between $\bq$ and $\bp$ (i.e. the fraction of the light
scattered at $\bq$ towards $\bp$ that reaches $\bp$).</li>
</ul>

<p>Thus, by noting $\bw_s$ the unit direction vector towards the Sun, and with
the following definitions:
<ul>
<li>$r=\Vert\bo\bp\Vert$,</li>
<li>$d=\Vert\bp\bq\Vert$,</li>
<li>$\mu=(\bo\bp\cdot\bp\bq)/rd$,</li>
<li>$\mu_s=(\bo\bp\cdot\bw_s)/r$,</li>
<li>$\nu=(\bp\bq\cdot\bw_s)/d$</li>
</ul>
the values of $r$ and $\mu_s$ for $\bq$ are
<ul>
<li>$r_d=\Vert\bo\bq\Vert=\sqrt{d^2+2r\mu d +r^2}$,</li>
<li>$\mu_{s,d}=(\bo\bq\cdot\bw_s)/r_d=((\bo\bp+\bp\bq)\cdot\bw_s)/r_d=
(r\mu_s + d\nu)/r_d$</li>
</ul>
and the Rayleigh and Mie single scattering components can be computed as follows
(note that we omit the solar irradiance and the phase function terms, as well as
the scattering coefficients at the bottom of the atmosphere - we add them later
on for efficiency reasons):
*/

void ComputeSingleScatteringIntegrand(IN(AtmosphereComponents) atmosphere,
    IN(sampler2D) transmittance_texture, float r, float mu, float mu_s, float nu, float d,
    bool ray_r_mu_intersects_ground, OUT(vec3) molecules, OUT(vec3) aerosols) {
  float r_d    = ClampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
  float mu_s_d = ClampCosine((r * mu_s + d * nu) / r_d);
  vec3  transmittance =
      GetTransmittance(transmittance_texture, r, mu, d, ray_r_mu_intersects_ground) *
      GetTransmittanceToSun(transmittance_texture, r_d, mu_s_d);
  molecules = transmittance * GetDensity(atmosphere.molecules.densityTextureV, r_d - BOTTOM_RADIUS);
  aerosols  = transmittance * GetDensity(atmosphere.aerosols.densityTextureV, r_d - BOTTOM_RADIUS);
}

/*
<p>Consider now the Sun light arriving at $\bp$ from a given direction $\bw$,
after exactly one scattering event. The scattering event can occur at any point
$\bq$ between $\bp$ and the intersection $\bi$ of the half-line $[\bp,\bw)$ with
the nearest atmosphere boundary. Thus, the single scattered radiance at $\bp$
from direction $\bw$ is the integral of the single scattered radiance from $\bq$
to $\bp$ for all points $\bq$ between $\bp$ and $\bi$. To compute it, we first
need the length $\Vert\bp\bi\Vert$:
*/

float DistanceToNearestAtmosphereBoundary(float r, float mu, bool ray_r_mu_intersects_ground) {
  if (ray_r_mu_intersects_ground) {
    return DistanceToBottomAtmosphereBoundary(r, mu);
  } else {
    return DistanceToTopAtmosphereBoundary(r, mu);
  }
}

/*
<p>The single scattering integral can then be computed as follows (using
the <a href="https://en.wikipedia.org/wiki/Trapezoidal_rule">trapezoidal
rule</a>):
*/

void ComputeSingleScattering(IN(AtmosphereComponents) atmosphere,
    IN(sampler2D) transmittance_texture, float r, float mu, float mu_s, float nu,
    bool ray_r_mu_intersects_ground, OUT(vec3) molecules, OUT(vec3) aerosols) {
  assert(r >= BOTTOM_RADIUS && r <= TOP_RADIUS);
  assert(mu >= -1.0 && mu <= 1.0);
  assert(mu_s >= -1.0 && mu_s <= 1.0);
  assert(nu >= -1.0 && nu <= 1.0);

  // The integration step, i.e. the length of each integration interval.
  float dx = DistanceToNearestAtmosphereBoundary(r, mu, ray_r_mu_intersects_ground) /
             float(SAMPLE_COUNT_SINGLE_SCATTERING);
  // Integration loop.
  vec3 molecules_sum = vec3(0.0);
  vec3 aerosols_sum  = vec3(0.0);
  for (int i = 0; i <= SAMPLE_COUNT_SINGLE_SCATTERING; ++i) {
    float d_i = float(i) * dx;
    // The Rayleigh and Mie single scattering at the current sample point.
    vec3 molecules_i;
    vec3 aerosols_i;
    ComputeSingleScatteringIntegrand(atmosphere, transmittance_texture, r, mu, mu_s, nu, d_i,
        ray_r_mu_intersects_ground, molecules_i, aerosols_i);
    // Sample weight (from the trapezoidal rule).
    float weight_i = (i == 0 || i == SAMPLE_COUNT_SINGLE_SCATTERING) ? 0.5 : 1.0;
    molecules_sum += molecules_i * weight_i;
    aerosols_sum += aerosols_i * weight_i;
  }
  molecules = molecules_sum * dx * SOLAR_IRRADIANCE * atmosphere.molecules.scattering;
  aerosols  = aerosols_sum * dx * SOLAR_IRRADIANCE * atmosphere.aerosols.scattering;
}

vec3 PhaseFunction(ScatteringComponent component, float nu) {
  float theta = acos(nu) / PI; // 0<->1
  return texture2D(phase_texture, vec2(theta, component.phaseTextureV)).rgb;
}

/*
<h4 id="single_scattering_precomputation">Precomputation</h4>

<p>The <code>ComputeSingleScattering</code> function is quite costly to
evaluate, and a lot of evaluations are needed to compute multiple scattering.
We therefore want to precompute it in a texture, which requires a mapping from
the 4 function parameters to texture coordinates. Assuming for now that we have
4D textures, we need to define a mapping from $(r,\mu,\mu_s,\nu)$ to texture
coordinates $(u,v,w,z)$. The function below implements the mapping defined in
our <a href="https://hal.inria.fr/inria-00288758/en">paper</a>, with some small
improvements (refer to the paper and to the above figures for the notations):
<ul>
<li>the mapping for $\mu$ takes the minimal distance to the nearest atmosphere
boundary into account, to map $\mu$ to the full $[0,1]$ interval (the original
mapping was not covering the full $[0,1]$ interval).</li>
<li>the mapping for $\mu_s$ is more generic than in the paper (the original
mapping was using ad-hoc constants chosen for the Earth atmosphere case). It is
based on the distance to the top atmosphere boundary (for the sun rays), as for
the $\mu$ mapping, and uses only one ad-hoc (but configurable) parameter. Yet,
as the original definition, it provides an increased sampling rate near the
horizon.</li>
</ul>
*/

vec4 GetScatteringTextureUvwzFromRMuMuSNu(
    float r, float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground) {
  assert(r >= BOTTOM_RADIUS && r <= TOP_RADIUS);
  assert(mu >= -1.0 && mu <= 1.0);
  assert(mu_s >= -1.0 && mu_s <= 1.0);
  assert(nu >= -1.0 && nu <= 1.0);

  // Distance to top atmosphere boundary for a horizontal ray at ground level.
  float H = sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the horizon.
  float rho = SafeSqrt(r * r - BOTTOM_RADIUS * BOTTOM_RADIUS);
  float u_r = GetTextureCoordFromUnitRange(rho / H, SCATTERING_TEXTURE_R_SIZE);

  // Discriminant of the quadratic equation for the intersections of the ray
  // (r,mu) with the ground (see RayIntersectsGround).
  float r_mu         = r * mu;
  float discriminant = r_mu * r_mu - r * r + BOTTOM_RADIUS * BOTTOM_RADIUS;
  float u_mu;
  if (ray_r_mu_intersects_ground) {
    // Distance to the ground for the ray (r,mu), and its minimum and maximum
    // values over all mu - obtained for (r,-1) and (r,mu_horizon).
    float d     = -r_mu - SafeSqrt(discriminant);
    float d_min = r - BOTTOM_RADIUS;
    float d_max = rho;
    u_mu        = 0.5 -
           0.5 * GetTextureCoordFromUnitRange(d_max == d_min ? 0.0 : (d - d_min) / (d_max - d_min),
                     SCATTERING_TEXTURE_MU_SIZE / 2);
  } else {
    // Distance to the top atmosphere boundary for the ray (r,mu), and its
    // minimum and maximum values over all mu - obtained for (r,1) and
    // (r,mu_horizon).
    float d     = -r_mu + SafeSqrt(discriminant + H * H);
    float d_min = TOP_RADIUS - r;
    float d_max = rho + H;
    u_mu        = 0.5 + 0.5 * GetTextureCoordFromUnitRange(
                           (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
  }

  float d     = DistanceToTopAtmosphereBoundary(BOTTOM_RADIUS, mu_s);
  float d_min = TOP_RADIUS - BOTTOM_RADIUS;
  float d_max = H;
  float a     = (d - d_min) / (d_max - d_min);
  float D     = DistanceToTopAtmosphereBoundary(BOTTOM_RADIUS, MU_S_MIN);
  float A     = (D - d_min) / (d_max - d_min);
  // An ad-hoc function equal to 0 for mu_s = MU_S_MIN (because then d = D and
  // thus a = A), equal to 1 for mu_s = 1 (because then d = d_min and thus
  // a = 0), and with a large slope around mu_s = 0, to get more texture
  // samples near the horizon.
  float u_mu_s =
      GetTextureCoordFromUnitRange(max(1.0 - a / A, 0.0) / (1.0 + a), SCATTERING_TEXTURE_MU_S_SIZE);

  float u_nu = (nu + 1.0) / 2.0;
  return vec4(u_nu, u_mu_s, u_mu, u_r);
}

/*
<p>The inverse mapping follows immediately:
*/

void GetRMuMuSNuFromScatteringTextureUvwz(IN(vec4) uvwz, OUT(float) r, OUT(float) mu,
    OUT(float) mu_s, OUT(float) nu, OUT(bool) ray_r_mu_intersects_ground) {
  assert(uvwz.x >= 0.0 && uvwz.x <= 1.0);
  assert(uvwz.y >= 0.0 && uvwz.y <= 1.0);
  assert(uvwz.z >= 0.0 && uvwz.z <= 1.0);
  assert(uvwz.w >= 0.0 && uvwz.w <= 1.0);

  // Distance to top atmosphere boundary for a horizontal ray at ground level.
  float H = sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS);
  // Distance to the horizon.
  float rho = H * GetUnitRangeFromTextureCoord(uvwz.w, SCATTERING_TEXTURE_R_SIZE);
  r         = sqrt(rho * rho + BOTTOM_RADIUS * BOTTOM_RADIUS);

  if (uvwz.z < 0.5) {
    // Distance to the ground for the ray (r,mu), and its minimum and maximum
    // values over all mu - obtained for (r,-1) and (r,mu_horizon) - from which
    // we can recover mu:
    float d_min = r - BOTTOM_RADIUS;
    float d_max = rho;
    float d     = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
                                            1.0 - 2.0 * uvwz.z, SCATTERING_TEXTURE_MU_SIZE / 2);
    mu = d == 0.0 ? -1.0 : ClampCosine(-(rho * rho + d * d) / (2.0 * r * d));
    ray_r_mu_intersects_ground = true;
  } else {
    // Distance to the top atmosphere boundary for the ray (r,mu), and its
    // minimum and maximum values over all mu - obtained for (r,1) and
    // (r,mu_horizon) - from which we can recover mu:
    float d_min = TOP_RADIUS - r;
    float d_max = rho + H;
    float d     = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
                                            2.0 * uvwz.z - 1.0, SCATTERING_TEXTURE_MU_SIZE / 2);
    mu = d == 0.0 ? 1.0 : ClampCosine((H * H - rho * rho - d * d) / (2.0 * r * d));
    ray_r_mu_intersects_ground = false;
  }

  float x_mu_s = GetUnitRangeFromTextureCoord(uvwz.y, SCATTERING_TEXTURE_MU_S_SIZE);
  float d_min  = TOP_RADIUS - BOTTOM_RADIUS;
  float d_max  = H;
  float D      = DistanceToTopAtmosphereBoundary(BOTTOM_RADIUS, MU_S_MIN);
  float A      = (D - d_min) / (d_max - d_min);
  float a      = (A - x_mu_s * A) / (1.0 + x_mu_s * A);
  float d      = d_min + min(a, A) * (d_max - d_min);
  mu_s         = d == 0.0 ? 1.0 : ClampCosine((H * H - d * d) / (2.0 * BOTTOM_RADIUS * d));

  nu = ClampCosine(uvwz.x * 2.0 - 1.0);
}

/*
<p>We assumed above that we have 4D textures, which is not the case in practice.
We therefore need a further mapping, between 3D and 4D texture coordinates. The
function below expands a 3D texel coordinate into a 4D texture coordinate, and
then to $(r,\mu,\mu_s,\nu)$ parameters. It does so by "unpacking" two texel
coordinates from the $x$ texel coordinate. Note also how we clamp the $\nu$
parameter at the end. This is because $\nu$ is not a fully independent variable:
its range of values depends on $\mu$ and $\mu_s$ (this can be seen by computing
$\mu$, $\mu_s$ and $\nu$ from the cartesian coordinates of the zenith, view and
sun unit direction vectors), and the previous functions implicitely assume this
(their assertions can break if this constraint is not respected).
*/

void GetRMuMuSNuFromScatteringTextureFragCoord(IN(vec3) frag_coord, OUT(float) r, OUT(float) mu,
    OUT(float) mu_s, OUT(float) nu, OUT(bool) ray_r_mu_intersects_ground) {
  const vec4 SCATTERING_TEXTURE_SIZE = vec4(SCATTERING_TEXTURE_NU_SIZE - 1,
      SCATTERING_TEXTURE_MU_S_SIZE, SCATTERING_TEXTURE_MU_SIZE, SCATTERING_TEXTURE_R_SIZE);
  float      frag_coord_nu           = floor(frag_coord.x / float(SCATTERING_TEXTURE_MU_S_SIZE));
  float      frag_coord_mu_s         = mod(frag_coord.x, float(SCATTERING_TEXTURE_MU_S_SIZE));
  vec4       uvwz =
      vec4(frag_coord_nu, frag_coord_mu_s, frag_coord.y, frag_coord.z) / SCATTERING_TEXTURE_SIZE;
  GetRMuMuSNuFromScatteringTextureUvwz(uvwz, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
  // Clamp nu to its valid range of values, given mu and mu_s.
  nu = clamp(nu, mu * mu_s - sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)),
      mu * mu_s + sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)));
}

/*
<p>With this mapping, we can finally write a function to precompute a texel of
the single scattering in a 3D texture:
*/

void ComputeSingleScatteringTexture(IN(AtmosphereComponents) atmosphere,
    IN(sampler2D) transmittance_texture, IN(vec3) frag_coord, OUT(vec3) molecules,
    OUT(vec3) aerosols) {
  float r;
  float mu;
  float mu_s;
  float nu;
  bool  ray_r_mu_intersects_ground;
  GetRMuMuSNuFromScatteringTextureFragCoord(
      frag_coord, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
  ComputeSingleScattering(atmosphere, transmittance_texture, r, mu, mu_s, nu,
      ray_r_mu_intersects_ground, molecules, aerosols);
}

/*
<h4 id="single_scattering_lookup">Lookup</h4>

<p>With the help of the above precomputed texture, we can now get the scattering
between a point and the nearest atmosphere boundary with two texture lookups (we
need two 3D texture lookups to emulate a single 4D texture lookup with
quadrilinear interpolation; the 3D texture coordinates are computed using the
inverse of the 3D-4D mapping defined in
<code>GetRMuMuSNuFromScatteringTextureFragCoord</code>):
*/

TEMPLATE(vec3)
vec3 GetScattering(IN(sampler3D TEMPLATE_ARGUMENT(vec3)) scattering_texture, float r, float mu,
    float mu_s, float nu, bool ray_r_mu_intersects_ground) {
  vec4  uvwz = GetScatteringTextureUvwzFromRMuMuSNu(r, mu, mu_s, nu, ray_r_mu_intersects_ground);
  float tex_coord_x = uvwz.x * float(SCATTERING_TEXTURE_NU_SIZE - 1);
  float tex_x       = floor(tex_coord_x);
  float lerp        = tex_coord_x - tex_x;
  vec3  uvw0        = vec3((tex_x + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);
  vec3  uvw1 = vec3((tex_x + 1.0 + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);
  return vec3(
      texture(scattering_texture, uvw0) * (1.0 - lerp) + texture(scattering_texture, uvw1) * lerp);
}

/*
<p>Finally, we provide here a convenience lookup function which will be useful
in the next section. This function returns either the single scattering, with
the phase functions included, or the $n$-th order of scattering, with $n>1$. It
assumes that, if <code>scattering_order</code> is strictly greater than 1, then
<code>multiple_scattering_texture</code> corresponds to this scattering order,
with both Rayleigh and Mie included, as well as all the phase function terms.
*/

vec3 GetScattering(IN(AtmosphereComponents) atmosphere,
    IN(sampler3D) single_molecules_scattering_texture,
    IN(sampler3D) single_aerosols_scattering_texture, IN(sampler3D) multiple_scattering_texture,
    float r, float mu, float mu_s, float nu, bool ray_r_mu_intersects_ground,
    int scattering_order) {
  if (scattering_order == 1) {
    vec3 molecules = GetScattering(
        single_molecules_scattering_texture, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    vec3 aerosols = GetScattering(
        single_aerosols_scattering_texture, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    return molecules * PhaseFunction(atmosphere.molecules, nu) +
           aerosols * PhaseFunction(atmosphere.aerosols, nu);
  } else {
    return GetScattering(multiple_scattering_texture, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
  }
}

/*
<h3 id="multiple_scattering">Multiple scattering</h3>

<p>The multiply scattered radiance is the light arriving from the Sun at some
point in the atmosphere after two or more <i>bounces</i> (where a bounce is
either a scattering event or a reflection from the ground). The following
sections describe how we compute it, how we store it in a precomputed texture,
and how we read it back.

<p>Note that, as for single scattering, we exclude here the light paths whose
<i>last</i> bounce is a reflection on the ground. The contribution from these
paths is computed separately, at rendering time, in order to take the actual
ground albedo into account (for intermediate reflections on the ground, which
are precomputed, we use an average, uniform albedo).

<h4 id="multiple_scattering_computation">Computation</h4>

<p>Multiple scattering can be decomposed into the sum of double scattering,
triple scattering, etc, where each term corresponds to the light arriving from
the Sun at some point in the atmosphere after <i>exactly</i> 2, 3, etc bounces.
Moreover, each term can be computed from the previous one. Indeed, the light
arriving at some point $\bp$ from direction $\bw$ after $n$ bounces is an
integral over all the possible points $\bq$ for the last bounce, which involves
the light arriving at $\bq$ from any direction, after $n-1$ bounces.

<p>This description shows that each scattering order requires a triple integral
to be computed from the previous one (one integral over all the points $\bq$
on the segment from $\bp$ to the nearest atmosphere boundary in direction $\bw$,
and a nested double integral over all directions at each point $\bq$).
Therefore, if we wanted to compute each order "from scratch", we would need a
triple integral for double scattering, a sextuple integral for triple
scattering, etc. This would be clearly inefficient, because of all the redundant
computations (the computations for order $n$ would basically redo all the
computations for all previous orders, leading to quadratic complexity in the
total float of orders). Instead, it is much more efficient to proceed as
follows:
<ul>
<li>precompute single scattering in a texture (as described above),</li>
<li>for $n \ge 2$:
<ul>
<li>precompute the $n$-th scattering in a texture, with a triple integral whose
integrand uses lookups in the $(n-1)$-th scattering texture</li>
</ul>
</li>
</ul>

<p>This strategy avoids many redundant computations but does not eliminate all
of them. Consider for instance the points $\bp$ and $\bp'$ in the figure below,
and the computations which are necessary to compute the light arriving at these
two points from direction $\bw$ after $n$ bounces. These computations involve,
in particular, the evaluation of the radiance $L$ which is scattered at $\bq$ in
direction $-\bw$, and coming from all directions after $n-1$ bounces:

<svg width="340px" height="150px">
  <style type="text/css"><![CDATA[
    circle { fill: #000000; stroke: none; }
    path { fill: none; stroke: #000000; }
    text { font-size: 16px; font-style: normal; font-family: Sans; }
    .vector { font-weight: bold; }
  ]]></style>
  <path d="m 0,26 a 600,600 0 0 1 340,0"/>
  <path d="m 0,110 a 520,520 0 0 1 340,0"/>
  <path d="m 170,140 0,-135"/>
  <path d="m 20,80 200,-40" />
  <path d="m 209,39 11,1 -10,5" />
  <circle cx="70" cy="70" r="2.5"/>
  <circle cx="120" cy="60" r="2.5"/>
  <circle cx="170" cy="50" r="2.5"/>
  <text x="65" y="60" class="vector">p</text>
  <text x="175" y="65" class="vector">q</text>
  <text x="225" y="35" class="vector">ω</text>
  <text x="115" y="50" class="vector">p<tspan
     style="font-weight:normal;">'</tspan></text>
</svg>

<p>Therefore, if we computed the n-th scattering with a triple integral as
described above, we would compute $L$ redundantly (in fact, for all points $\bp$
between $\bq$ and the nearest atmosphere boundary in direction $-\bw$). To avoid
this, and thus increase the efficiency of the multiple scattering computations,
we refine the above algorithm as follows:
<ul>
<li>precompute single scattering in a texture (as described above),</li>
<li>for $n \ge 2$:
<ul>
<li>for each point $\bq$ and direction $\bw$, precompute the light which is
scattered at $\bq$ towards direction $-\bw$, coming from any direction after
$n-1$ bounces (this involves only a double integral, whose integrand uses
lookups in the $(n-1)$-th scattering texture),</li>
<li>for each point $\bp$ and direction $\bw$, precompute the light coming from
direction $\bw$ after $n$ bounces (this involves only a single integral, whose
integrand uses lookups in the texture computed at the previous line)</li>
</ul>
</li>
</ul>

<p>To get a complete algorithm, we must now specify how we implement the two
steps in the above loop. This is what we do in the rest of this section.

<h5 id="multiple_scattering_first_step">First step</h5>

<p>The first step computes the radiance which is scattered at some point $\bq$
inside the atmosphere, towards some direction $-\bw$. Furthermore, we assume
that this scattering event is the $n$-th bounce.

<p>This radiance is the integral over all the possible incident directions
$\bw_i$, of the product of
<ul>
<li>the incident radiance $L_i$ arriving at $\bq$ from direction $\bw_i$ after
$n-1$ bounces, which is the sum of:
<ul>
<li>a term given by the precomputed scattering texture for the $(n-1)$-th
order,</li>
<li>if the ray $[\bq, \bw_i)$ intersects the ground at $\br$, the contribution
from the light paths with $n-1$ bounces and whose last bounce is at $\br$, i.e.
on the ground (these paths are excluded, by definition, from our precomputed
textures, but we must take them into account here since the bounce on the ground
is followed by a bounce at $\bq$). This contribution, in turn, is the product
of:
<ul>
<li>the transmittance between $\bq$ and $\br$,</li>
<li>the (average) ground albedo,</li>
<li>the <a href="https://www.cs.princeton.edu/~smr/cs348c-97/surveypaper.html"
>Lambertian BRDF</a> $1/\pi$,</li>
<li>the irradiance received on the ground after $n-2$ bounces. We explain in the
<a href="#irradiance">next section</a> how we precompute it in a texture. For
now, we assume that we can use the following function to retrieve this
irradiance from a precomputed texture:
</li>
</ul>
</li>
</ul>
</li>
</ul>
*/

vec3 GetIrradiance(IN(sampler2D) irradiance_texture, float r, float mu_s);

/*
<ul>
<li>the scattering coefficient at $\bq$,</li>
<li>the scattering phase function for the directions $\bw$ and $\bw_i$</li>
</ul>
This leads to the following implementation (where
<code>multiple_scattering_texture</code> is supposed to contain the $(n-1)$-th
order of scattering, if $n>2$, <code>irradiance_texture</code> is the irradiance
received on the ground after $n-2$ bounces, and <code>scattering_order</code> is
equal to $n$):</li>
*/

vec3 ComputeScatteringDensity(IN(AtmosphereComponents) atmosphere,
    IN(sampler2D) transmittance_texture, IN(sampler3D) single_molecules_scattering_texture,
    IN(sampler3D) single_aerosols_scattering_texture, IN(sampler3D) multiple_scattering_texture,
    IN(sampler2D) irradiance_texture, float r, float mu, float mu_s, float nu,
    int scattering_order) {
  assert(r >= BOTTOM_RADIUS && r <= TOP_RADIUS);
  assert(mu >= -1.0 && mu <= 1.0);
  assert(mu_s >= -1.0 && mu_s <= 1.0);
  assert(nu >= -1.0 && nu <= 1.0);
  assert(scattering_order >= 2);

  // Compute unit direction vectors for the zenith, the view direction omega and
  // and the sun direction omega_s, such that the cosine of the view-zenith
  // angle is mu, the cosine of the sun-zenith angle is mu_s, and the cosine of
  // the view-sun angle is nu. The goal is to simplify computations below.
  vec3  zenith_direction = vec3(0.0, 0.0, 1.0);
  vec3  omega            = vec3(sqrt(1.0 - mu * mu), 0.0, mu);
  float sun_dir_x        = omega.x == 0.0 ? 0.0 : (nu - mu * mu_s) / omega.x;
  float sun_dir_y        = sqrt(max(1.0 - sun_dir_x * sun_dir_x - mu_s * mu_s, 0.0));
  vec3  omega_s          = vec3(sun_dir_x, sun_dir_y, mu_s);

  const float dphi               = PI / float(SAMPLE_COUNT_SCATTERING_DENSITY);
  const float dtheta             = PI / float(SAMPLE_COUNT_SCATTERING_DENSITY);
  vec3        molecules_aerosols = vec3(0.0);

  // Nested loops for the integral over all the incident directions omega_i.
  for (int l = 0; l < SAMPLE_COUNT_SCATTERING_DENSITY; ++l) {
    float theta                         = (float(l) + 0.5) * dtheta;
    float cos_theta                     = cos(theta);
    float sin_theta                     = sin(theta);
    bool  ray_r_theta_intersects_ground = RayIntersectsGround(r, cos_theta);

    // The distance and transmittance to the ground only depend on theta, so we
    // can compute them in the outer loop for efficiency.
    float distance_to_ground      = 0.0;
    vec3  transmittance_to_ground = vec3(0.0);
    vec3  ground_albedo           = vec3(0.0);
    if (ray_r_theta_intersects_ground) {
      distance_to_ground      = DistanceToBottomAtmosphereBoundary(r, cos_theta);
      transmittance_to_ground = GetTransmittance(transmittance_texture, r, cos_theta,
          distance_to_ground, true /* ray_intersects_ground */);
      ground_albedo           = GROUND_ALBEDO;
    }

    for (int m = 0; m < 2 * SAMPLE_COUNT_SCATTERING_DENSITY; ++m) {
      float phi      = (float(m) + 0.5) * dphi;
      vec3  omega_i  = vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
      float domega_i = dtheta * dphi * sin(theta);

      // The radiance L_i arriving from direction omega_i after n-1 bounces is
      // the sum of a term given by the precomputed scattering texture for the
      // (n-1)-th order:
      float nu1               = dot(omega_s, omega_i);
      vec3  incident_radiance = GetScattering(atmosphere, single_molecules_scattering_texture,
          single_aerosols_scattering_texture, multiple_scattering_texture, r, omega_i.z, mu_s, nu1,
          ray_r_theta_intersects_ground, scattering_order - 1);

      // and of the contribution from the light paths with n-1 bounces and whose
      // last bounce is on the ground. This contribution is the product of the
      // transmittance to the ground, the ground albedo, the ground BRDF, and
      // the irradiance received on the ground after n-2 bounces.
      vec3 ground_normal = normalize(zenith_direction * r + omega_i * distance_to_ground);
      vec3 ground_irradiance =
          GetIrradiance(irradiance_texture, BOTTOM_RADIUS, dot(ground_normal, omega_s));
      incident_radiance += transmittance_to_ground * ground_albedo * (1.0 / PI) * ground_irradiance;

      // The radiance finally scattered from direction omega_i towards direction
      // -omega is the product of the incident radiance, the scattering
      // coefficient, and the phase function for directions omega and omega_i
      // (all this summed over all particle types, i.e. Rayleigh and Mie).
      float nu2               = dot(omega, omega_i);
      float molecules_density = GetDensity(atmosphere.molecules.densityTextureV, r - BOTTOM_RADIUS);
      float aerosols_density  = GetDensity(atmosphere.aerosols.densityTextureV, r - BOTTOM_RADIUS);
      molecules_aerosols += incident_radiance *
                            (atmosphere.molecules.scattering * molecules_density *
                                    PhaseFunction(atmosphere.molecules, nu2) +
                                atmosphere.aerosols.scattering * aerosols_density *
                                    PhaseFunction(atmosphere.aerosols, nu2)) *
                            domega_i;
    }
  }
  return molecules_aerosols;
}

/*
<h5 id="multiple_scattering_second_step">Second step</h5>

<p>The second step to compute the $n$-th order of scattering is to compute for
each point $\bp$ and direction $\bw$, the radiance coming from direction $\bw$
after $n$ bounces, using a texture precomputed with the previous function.

<p>This radiance is the integral over all points $\bq$ between $\bp$ and the
nearest atmosphere boundary in direction $\bw$ of the product of:
<ul>
<li>a term given by a texture precomputed with the previous function, namely
the radiance scattered at $\bq$ towards $\bp$, coming from any direction after
$n-1$ bounces,</li>
<li>the transmittance betweeen $\bp$ and $\bq$</li>
</ul>
Note that this excludes the light paths with $n$ bounces and whose last
bounce is on the ground, on purpose. Indeed, we chose to exclude these paths
from our precomputed textures so that we can compute them at render time
instead, using the actual ground albedo.

<p>The implementation for this second step is straightforward:
*/

vec3 ComputeMultipleScattering(IN(sampler2D) transmittance_texture,
    IN(sampler3D) scattering_density_texture, float r, float mu, float mu_s, float nu,
    bool ray_r_mu_intersects_ground) {
  assert(r >= BOTTOM_RADIUS && r <= TOP_RADIUS);
  assert(mu >= -1.0 && mu <= 1.0);
  assert(mu_s >= -1.0 && mu_s <= 1.0);
  assert(nu >= -1.0 && nu <= 1.0);

  // The integration step, i.e. the length of each integration interval.
  float dx = DistanceToNearestAtmosphereBoundary(r, mu, ray_r_mu_intersects_ground) /
             float(SAMPLE_COUNT_MULTI_SCATTERING);
  // Integration loop.
  vec3 molecules_aerosols_sum = vec3(0.0);
  for (int i = 0; i <= SAMPLE_COUNT_MULTI_SCATTERING; ++i) {
    float d_i = float(i) * dx;

    // The r, mu and mu_s parameters at the current integration point (see the
    // single scattering section for a detailed explanation).
    float r_i    = ClampRadius(sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r));
    float mu_i   = ClampCosine((r * mu + d_i) / r_i);
    float mu_s_i = ClampCosine((r * mu_s + d_i * nu) / r_i);

    // The Rayleigh and Mie multiple scattering at the current sample point.
    vec3 molecules_aerosols_i =
        GetScattering(
            scattering_density_texture, r_i, mu_i, mu_s_i, nu, ray_r_mu_intersects_ground) *
        GetTransmittance(transmittance_texture, r, mu, d_i, ray_r_mu_intersects_ground) * dx;
    // Sample weight (from the trapezoidal rule).
    float weight_i = (i == 0 || i == SAMPLE_COUNT_MULTI_SCATTERING) ? 0.5 : 1.0;
    molecules_aerosols_sum += molecules_aerosols_i * weight_i;
  }
  return molecules_aerosols_sum;
}

/*
<h4 id="multiple_scattering_precomputation">Precomputation</h4>

<p>As explained in the <a href="#multiple_scattering">overall algorithm</a> to
compute multiple scattering, we need to precompute each order of scattering in a
texture to save computations while computing the next order. And, in order to
store a function in a texture, we need a mapping from the function parameters to
texture coordinates. Fortunately, all the orders of scattering depend on the
same $(r,\mu,\mu_s,\nu)$ parameters as single scattering, so we can simple reuse
the mappings defined for single scattering. This immediately leads to the
following simple functions to precompute a texel of the textures for the
<a href="#multiple_scattering_first_step">first</a> and
<a href="#multiple_scattering_second_step">second</a> steps of each iteration
over the float of bounces:
*/

vec3 ComputeScatteringDensityTexture(IN(AtmosphereComponents) atmosphere,
    IN(sampler2D) transmittance_texture, IN(sampler3D) single_molecules_scattering_texture,
    IN(sampler3D) single_aerosols_scattering_texture, IN(sampler3D) multiple_scattering_texture,
    IN(sampler2D) irradiance_texture, IN(vec3) frag_coord, int scattering_order) {
  float r;
  float mu;
  float mu_s;
  float nu;
  bool  ray_r_mu_intersects_ground;
  GetRMuMuSNuFromScatteringTextureFragCoord(
      frag_coord, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
  return ComputeScatteringDensity(atmosphere, transmittance_texture,
      single_molecules_scattering_texture, single_aerosols_scattering_texture,
      multiple_scattering_texture, irradiance_texture, r, mu, mu_s, nu, scattering_order);
}

vec3 ComputeMultipleScatteringTexture(IN(sampler2D) transmittance_texture,
    IN(sampler3D) scattering_density_texture, IN(vec3) frag_coord, OUT(float) nu) {
  float r;
  float mu;
  float mu_s;
  bool  ray_r_mu_intersects_ground;
  GetRMuMuSNuFromScatteringTextureFragCoord(
      frag_coord, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
  return ComputeMultipleScattering(transmittance_texture, scattering_density_texture, r, mu, mu_s,
      nu, ray_r_mu_intersects_ground);
}

/*
<h4 id="multiple_scattering_lookup">Lookup</h4>

<p>Likewise, we can simply reuse the lookup function <code>GetScattering</code>
implemented for single scattering to read a value from the precomputed textures
for multiple scattering. In fact, this is what we did above in the
<code>ComputeScatteringDensity</code> and <code>ComputeMultipleScattering</code>
functions.

<h3 id="irradiance">Ground irradiance</h3>

<p>The ground irradiance is the Sun light received on the ground after $n \ge 0$
bounces (where a bounce is either a scattering event or a reflection on the
ground). We need this for two purposes:
<ul>
<li>while precomputing the $n$-th order of scattering, with $n \ge 2$, in order
to compute the contribution of light paths whose $(n-1)$-th bounce is on the
ground (which requires the ground irradiance after $n-2$ bounces - see the
<a href="#multiple_scattering_computation">Multiple scattering</a>
section),</li>
<li>at rendering time, to compute the contribution of light paths whose last
bounce is on the ground (these paths are excluded, by definition, from our
precomputed scattering textures)</li>
</ul>

<p>In the first case we only need the ground irradiance for horizontal surfaces
at the bottom of the atmosphere (during precomputations we assume a perfectly
spherical ground with a uniform albedo). In the second case, however, we need
the ground irradiance for any altitude and any surface normal, and we want to
precompute it for efficiency. In fact, as described in our
<a href="https://hal.inria.fr/inria-00288758/en">paper</a> we precompute it only
for horizontal surfaces, at any altitude (which requires only 2D textures,
instead of 4D textures for the general case), and we use approximations for
non-horizontal surfaces.

<p>The following sections describe how we compute the ground irradiance, how we
store it in a precomputed texture, and how we read it back.

<h4 id="irradiance_computation">Computation</h4>

<p>The ground irradiance computation is different for the direct irradiance,
i.e. the light received directly from the Sun, without any intermediate bounce,
and for the indirect irradiance (at least one bounce). We start here with the
direct irradiance.

<p>The irradiance is the integral over an hemisphere of the incident radiance,
times a cosine factor. For the direct ground irradiance, the incident radiance
is the Sun radiance at the top of the atmosphere, times the transmittance
through the atmosphere. And, since the Sun solid angle is small, we can
approximate the transmittance with a constant, i.e. we can move it outside the
irradiance integral, which can be performed over (the visible fraction of) the
Sun disc rather than the hemisphere. Then the integral becomes equivalent to the
ambient occlusion due to a sphere, also called a view factor, which is given in
<a href="http://webserver.dmt.upm.es/~isidoro/tc3/Radiation%20View%20factors.pdf
">Radiative view factors</a> (page 10). For a small solid angle, these complex
equations can be simplified as follows:
*/

vec3 ComputeDirectIrradiance(IN(sampler2D) transmittance_texture, float r, float mu_s) {
  assert(r >= BOTTOM_RADIUS && r <= TOP_RADIUS);
  assert(mu_s >= -1.0 && mu_s <= 1.0);

  float alpha_s = SUN_ANGULAR_RADIUS;
  // Approximate average of the cosine factor mu_s over the visible fraction of
  // the Sun disc.
  float average_cosine_factor =
      mu_s < -alpha_s
          ? 0.0
          : (mu_s > alpha_s ? mu_s : (mu_s + alpha_s) * (mu_s + alpha_s) / (4.0 * alpha_s));

  return SOLAR_IRRADIANCE *
         GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r, mu_s) *
         average_cosine_factor;
}

/*
<p>For the indirect ground irradiance the integral over the hemisphere must be
computed numerically. More precisely we need to compute the integral over all
the directions $\bw$ of the hemisphere, of the product of:
<ul>
<li>the radiance arriving from direction $\bw$ after $n$ bounces,
<li>the cosine factor, i.e. $\omega_z$</li>
</ul>
This leads to the following implementation (where
<code>multiple_scattering_texture</code> is supposed to contain the $n$-th
order of scattering, if $n>1$, and <code>scattering_order</code> is equal to
$n$):</li>
*/

vec3 ComputeIndirectIrradiance(IN(AtmosphereComponents) atmosphere,
    IN(sampler3D) single_molecules_scattering_texture,
    IN(sampler3D) single_aerosols_scattering_texture, IN(sampler3D) multiple_scattering_texture,
    float r, float mu_s, int scattering_order) {
  assert(r >= BOTTOM_RADIUS && r <= TOP_RADIUS);
  assert(mu_s >= -1.0 && mu_s <= 1.0);
  assert(scattering_order >= 1);

  const float dphi   = PI / float(SAMPLE_COUNT_INDIRECT_IRRADIANCE);
  const float dtheta = PI / float(SAMPLE_COUNT_INDIRECT_IRRADIANCE);

  vec3 result  = vec3(0.0);
  vec3 omega_s = vec3(sqrt(1.0 - mu_s * mu_s), 0.0, mu_s);
  for (int j = 0; j < SAMPLE_COUNT_INDIRECT_IRRADIANCE / 2; ++j) {
    float theta = (float(j) + 0.5) * dtheta;
    for (int i = 0; i < 2 * SAMPLE_COUNT_INDIRECT_IRRADIANCE; ++i) {
      float phi    = (float(i) + 0.5) * dphi;
      vec3  omega  = vec3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
      float domega = dtheta * dphi * sin(theta);

      float nu = dot(omega, omega_s);
      result += GetScattering(atmosphere, single_molecules_scattering_texture,
                    single_aerosols_scattering_texture, multiple_scattering_texture, r, omega.z,
                    mu_s, nu, false /* ray_r_theta_intersects_ground */, scattering_order) *
                omega.z * domega;
    }
  }
  return result;
}

/*
<h4 id="irradiance_precomputation">Precomputation</h4>

<p>In order to precompute the ground irradiance in a texture we need a mapping
from the ground irradiance parameters to texture coordinates. Since we
precompute the ground irradiance only for horizontal surfaces, this irradiance
depends only on $r$ and $\mu_s$, so we need a mapping from $(r,\mu_s)$ to
$(u,v)$ texture coordinates. The simplest, affine mapping is sufficient here,
because the ground irradiance function is very smooth:
*/

vec2 GetIrradianceTextureUvFromRMuS(float r, float mu_s) {
  assert(r >= BOTTOM_RADIUS && r <= TOP_RADIUS);
  assert(mu_s >= -1.0 && mu_s <= 1.0);
  float x_r    = (r - BOTTOM_RADIUS) / (TOP_RADIUS - BOTTOM_RADIUS);
  float x_mu_s = mu_s * 0.5 + 0.5;
  return vec2(GetTextureCoordFromUnitRange(x_mu_s, IRRADIANCE_TEXTURE_WIDTH),
      GetTextureCoordFromUnitRange(x_r, IRRADIANCE_TEXTURE_HEIGHT));
}

/*
<p>The inverse mapping follows immediately:
*/

void GetRMuSFromIrradianceTextureUv(IN(vec2) uv, OUT(float) r, OUT(float) mu_s) {
  assert(uv.x >= 0.0 && uv.x <= 1.0);
  assert(uv.y >= 0.0 && uv.y <= 1.0);
  float x_mu_s = GetUnitRangeFromTextureCoord(uv.x, IRRADIANCE_TEXTURE_WIDTH);
  float x_r    = GetUnitRangeFromTextureCoord(uv.y, IRRADIANCE_TEXTURE_HEIGHT);
  r            = BOTTOM_RADIUS + x_r * (TOP_RADIUS - BOTTOM_RADIUS);
  mu_s         = ClampCosine(2.0 * x_mu_s - 1.0);
}

/*
<p>It is now easy to define a fragment shader function to precompute a texel of
the ground irradiance texture, for the direct irradiance:
*/

const vec2 IRRADIANCE_TEXTURE_SIZE = vec2(IRRADIANCE_TEXTURE_WIDTH, IRRADIANCE_TEXTURE_HEIGHT);

vec3 ComputeDirectIrradianceTexture(IN(sampler2D) transmittance_texture, IN(vec2) frag_coord) {
  float r;
  float mu_s;
  GetRMuSFromIrradianceTextureUv(frag_coord / IRRADIANCE_TEXTURE_SIZE, r, mu_s);
  return ComputeDirectIrradiance(transmittance_texture, r, mu_s);
}

/*
<p>and the indirect one:
*/

vec3 ComputeIndirectIrradianceTexture(IN(AtmosphereComponents) atmosphere,
    IN(sampler3D) single_molecules_scattering_texture,
    IN(sampler3D) single_aerosols_scattering_texture, IN(sampler3D) multiple_scattering_texture,
    IN(vec2) frag_coord, int scattering_order) {
  float r;
  float mu_s;
  GetRMuSFromIrradianceTextureUv(frag_coord / IRRADIANCE_TEXTURE_SIZE, r, mu_s);
  return ComputeIndirectIrradiance(atmosphere, single_molecules_scattering_texture,
      single_aerosols_scattering_texture, multiple_scattering_texture, r, mu_s, scattering_order);
}

/*
<h4 id="irradiance_lookup">Lookup</h4>

<p>Thanks to these precomputed textures, we can now get the ground irradiance
with a single texture lookup:
*/

vec3 GetIrradiance(IN(sampler2D) irradiance_texture, float r, float mu_s) {
  vec2 uv = GetIrradianceTextureUvFromRMuS(r, mu_s);
  return vec3(texture(irradiance_texture, uv));
}

/*
<h3 id="rendering">Rendering</h3>

<p>Here we assume that the transmittance, scattering and irradiance textures
have been precomputed, and we provide functions using them to compute the sky
color, the aerial perspective, and the ground radiance.

<p>More precisely, we assume that the single Rayleigh scattering, without its
phase function term, plus the multiple scattering terms (divided by the Rayleigh
phase function for dimensional homogeneity) are stored in a
<code>multiple_scattering_texture</code>. We also assume that the single Mie scattering
is stored, without its phase function term in a <code>single_aerosols_scattering_texture</code>
*/

/*
<p>We can then retrieve all the scattering components (Rayleigh + multiple
scattering on one side, and single Mie scattering on the other side) with the
following function, based on
<a href="#single_scattering_lookup"><code>GetScattering</code></a> (we duplicate
some code here, instead of using two calls to <code>GetScattering</code>, to
make sure that the texture coordinates computation is shared between the lookups
in <code>multiple_scattering_texture</code> and
<code>single_aerosols_scattering_texture</code>):
*/

void GetCombinedScattering(IN(sampler3D) multiple_scattering_texture,
    IN(sampler3D) single_aerosols_scattering_texture, float r, float mu, float mu_s, float nu,
    bool ray_r_mu_intersects_ground, OUT(vec3) multiple_scattering,
    OUT(vec3) single_aerosols_scattering) {
  vec4  uvwz = GetScatteringTextureUvwzFromRMuMuSNu(r, mu, mu_s, nu, ray_r_mu_intersects_ground);
  float tex_coord_x = uvwz.x * float(SCATTERING_TEXTURE_NU_SIZE - 1);
  float tex_x       = floor(tex_coord_x);
  float lerp        = tex_coord_x - tex_x;
  vec3  uvw0        = vec3((tex_x + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);
  vec3  uvw1 = vec3((tex_x + 1.0 + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);

  multiple_scattering = vec3(texture(multiple_scattering_texture, uvw0) * (1.0 - lerp) +
                             texture(multiple_scattering_texture, uvw1) * lerp);
  single_aerosols_scattering =
      vec3(texture(single_aerosols_scattering_texture, uvw0) * (1.0 - lerp) +
           texture(single_aerosols_scattering_texture, uvw1) * lerp);
}

/*
<h4 id="rendering_sky">Sky</h4>

<p>To render the sky we simply need to display the sky radiance, which we can
get with a lookup in the precomputed scattering texture(s), multiplied by the
phase function terms that were omitted during precomputation. We can also return
the transmittance of the atmosphere (which we can get with a single lookup in
the precomputed transmittance texture), which is needed to correctly render the
objects in space (such as the Sun and the Moon). This leads to the following
function, where most of the computations are used to correctly handle the case
of viewers outside the atmosphere, and the case of light shafts:
*/

vec3 GetSkyRadiance(IN(AtmosphereComponents) atmosphere, IN(sampler2D) transmittance_texture,
    IN(sampler3D) multiple_scattering_texture, IN(sampler3D) single_aerosols_scattering_texture,
    vec3 camera, IN(vec3) view_ray, float shadow_length, IN(vec3) sun_direction,
    OUT(vec3) transmittance) {
  // Compute the distance to the top atmosphere boundary along the view ray,
  // assuming the viewer is in space (or NaN if the view ray does not intersect
  // the atmosphere).
  float r   = length(camera);
  float rmu = dot(camera, view_ray);
  float distance_to_top_atmosphere_boundary =
      -rmu - sqrt(rmu * rmu - r * r + TOP_RADIUS * TOP_RADIUS);
  // If the viewer is in space and the view ray intersects the atmosphere, move
  // the viewer to the top atmosphere boundary (along the view ray):
  if (distance_to_top_atmosphere_boundary > 0.0) {
    camera = camera + view_ray * distance_to_top_atmosphere_boundary;
    r      = TOP_RADIUS;
    rmu += distance_to_top_atmosphere_boundary;
  } else if (r > TOP_RADIUS) {
    // If the view ray does not intersect the atmosphere, simply return 0.
    transmittance = vec3(1.0);
    return vec3(0.0);
  }
  // Compute the r, mu, mu_s and nu parameters needed for the texture lookups.
  float mu                         = rmu / r;
  float mu_s                       = dot(camera, sun_direction) / r;
  float nu                         = dot(view_ray, sun_direction);
  bool  ray_r_mu_intersects_ground = RayIntersectsGround(r, mu);

  transmittance = ray_r_mu_intersects_ground
                      ? vec3(0.0)
                      : GetTransmittanceToTopAtmosphereBoundary(transmittance_texture, r, mu);
  vec3 multiple_scattering;
  vec3 single_aerosols_scattering;
  if (shadow_length == 0.0) {
    GetCombinedScattering(multiple_scattering_texture, single_aerosols_scattering_texture, r, mu,
        mu_s, nu, ray_r_mu_intersects_ground, multiple_scattering, single_aerosols_scattering);
  } else {
    // Case of light shafts (shadow_length is the total length noted l in our
    // paper): we omit the scattering between the camera and the point at
    // distance l, by implementing Eq. (18) of the paper (shadow_transmittance
    // is the T(x,x_s) term, scattering is the S|x_s=x+lv term).
    float d      = shadow_length;
    float r_p    = ClampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
    float mu_p   = (r * mu + d) / r_p;
    float mu_s_p = (r * mu_s + d * nu) / r_p;

    GetCombinedScattering(multiple_scattering_texture, single_aerosols_scattering_texture, r_p,
        mu_p, mu_s_p, nu, ray_r_mu_intersects_ground, multiple_scattering,
        single_aerosols_scattering);
    vec3 shadow_transmittance =
        GetTransmittance(transmittance_texture, r, mu, shadow_length, ray_r_mu_intersects_ground);
    multiple_scattering        = multiple_scattering * shadow_transmittance;
    single_aerosols_scattering = single_aerosols_scattering * shadow_transmittance;
  }
  return multiple_scattering * PhaseFunction(atmosphere.molecules, nu) +
         single_aerosols_scattering * PhaseFunction(atmosphere.aerosols, nu);
}

/*
<h4 id="rendering_aerial_perspective">Aerial perspective</h4>

<p>To render the aerial perspective we need the transmittance and the scattering
between two points (i.e. between the viewer and a point on the ground, which can
at an arbibrary altitude). We already have a function to compute the
transmittance between two points (using 2 lookups in a texture which only
contains the transmittance to the top of the atmosphere), but we don't have one
for the scattering between 2 points. Hopefully, the scattering between 2 points
can be computed from two lookups in a texture which contains the scattering to
the nearest atmosphere boundary, as for the transmittance (except that here the
two lookup results must be subtracted, instead of divided). This is what we
implement in the following function (the initial computations are used to
correctly handle the case of viewers outside the atmosphere):
*/

vec3 GetSkyRadianceToPoint(IN(AtmosphereComponents) atmosphere, IN(sampler2D) transmittance_texture,
    IN(sampler3D) multiple_scattering_texture, IN(sampler3D) single_aerosols_scattering_texture,
    vec3 camera, IN(vec3) point, float shadow_length, IN(vec3) sun_direction,
    OUT(vec3) transmittance) {
  // Compute the distance to the top atmosphere boundary along the view ray,
  // assuming the viewer is in space (or NaN if the view ray does not intersect
  // the atmosphere).
  vec3  view_ray = normalize(point - camera);
  float r        = length(camera);
  float rmu      = dot(camera, view_ray);
  float distance_to_top_atmosphere_boundary =
      -rmu - sqrt(rmu * rmu - r * r + TOP_RADIUS * TOP_RADIUS);
  // If the viewer is in space and the view ray intersects the atmosphere, move
  // the viewer to the top atmosphere boundary (along the view ray):
  if (distance_to_top_atmosphere_boundary > 0.0) {
    camera = camera + view_ray * distance_to_top_atmosphere_boundary;
    r      = TOP_RADIUS;
    rmu += distance_to_top_atmosphere_boundary;
  }

  // Compute the r, mu, mu_s and nu parameters for the first texture lookup.
  float mu                         = rmu / r;
  float mu_s                       = dot(camera, sun_direction) / r;
  float nu                         = dot(view_ray, sun_direction);
  float d                          = length(point - camera);
  bool  ray_r_mu_intersects_ground = RayIntersectsGround(r, mu);

  transmittance = GetTransmittance(transmittance_texture, r, mu, d, ray_r_mu_intersects_ground);

  vec3 multiple_scattering;
  vec3 single_aerosols_scattering;

  GetCombinedScattering(multiple_scattering_texture, single_aerosols_scattering_texture, r, mu,
      mu_s, nu, ray_r_mu_intersects_ground, multiple_scattering, single_aerosols_scattering);

  // Compute the r, mu, mu_s and nu parameters for the second texture lookup.
  // If shadow_length is not 0 (case of light shafts), we want to ignore the
  // scattering along the last shadow_length meters of the view ray, which we
  // do by subtracting shadow_length from d (this way multiple_scattering_p is equal to
  // the S|x_s=x_0-lv term in Eq. (17) of our paper).
  d            = max(d - shadow_length, 0.0);
  float r_p    = ClampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
  float mu_p   = (r * mu + d) / r_p;
  float mu_s_p = (r * mu_s + d * nu) / r_p;

  vec3 multiple_scattering_p;
  vec3 single_aerosols_scattering_p;

  GetCombinedScattering(multiple_scattering_texture, single_aerosols_scattering_texture, r_p, mu_p,
      mu_s_p, nu, ray_r_mu_intersects_ground, multiple_scattering_p, single_aerosols_scattering_p);

  // Combine the lookup results to get the scattering between camera and point.
  vec3 shadow_transmittance = transmittance;
  if (shadow_length > 0.0) {
    // This is the T(x,x_s) term in Eq. (17) of our paper, for light shafts.
    shadow_transmittance =
        GetTransmittance(transmittance_texture, r, mu, d, ray_r_mu_intersects_ground);
  }
  multiple_scattering = multiple_scattering - shadow_transmittance * multiple_scattering_p;
  single_aerosols_scattering =
      single_aerosols_scattering - shadow_transmittance * single_aerosols_scattering_p;

  // Hack to avoid rendering artifacts when the sun is below the horizon.
  // single_aerosols_scattering = single_aerosols_scattering * smoothstep(0.0, 0.01, mu_s);

  return multiple_scattering * PhaseFunction(atmosphere.molecules, nu) +
         single_aerosols_scattering * PhaseFunction(atmosphere.aerosols, nu);
}

/*
<h4 id="rendering_ground">Ground</h4>

<p>To render the ground we need the irradiance received on the ground after 0 or
more bounce(s) in the atmosphere or on the ground. The direct irradiance can be
computed with a lookup in the transmittance texture,
via <code>GetTransmittanceToSun</code>, while the indirect irradiance is given
by a lookup in the precomputed irradiance texture (this texture only contains
the irradiance for horizontal surfaces; we use the approximation defined in our
<a href="https://hal.inria.fr/inria-00288758/en">paper</a> for the other cases).
The function below returns the direct and indirect irradiances separately:
*/

vec3 GetSunAndSkyIrradiance(IN(sampler2D) transmittance_texture, IN(sampler2D) irradiance_texture,
    IN(vec3) point, IN(vec3) sun_direction, OUT(vec3) sky_irradiance) {
  float r    = length(point);
  float mu_s = dot(point, sun_direction) / r;

  // Indirect irradiance (approximated if the surface is not horizontal).
  sky_irradiance = GetIrradiance(irradiance_texture, r, mu_s);

  // Direct irradiance.
  return SOLAR_IRRADIANCE * GetTransmittanceToSun(transmittance_texture, r, mu_s);
}
