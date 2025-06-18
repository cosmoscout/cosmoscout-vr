////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// Lambertian reflectance to represent ideal diffuse surfaces.

// rho: Reflectivity of the surface in range [0, 1].

// Some suggestions for some bodies:
// - Moon: rho = 0.11

float $BRDF(vec3 N, vec3 L, vec3 V)
{
  return $rho / 3.14159265358979323846;
}