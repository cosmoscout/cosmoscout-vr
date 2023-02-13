////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ATMOSPHERES_MODEL_BASE_HPP
#define CSP_ATMOSPHERES_MODEL_BASE_HPP

#include <GL/glew.h>
#include <nlohmann/json.hpp>

namespace csp::atmospheres {

/// Any new atmospheric model must be derived from this class. Any preprocessing should be done in
/// init(). This method will also create a fragment shader containing the shader code required to
/// evaluate the model. The fragment shader should contain at least three methods:
///
///    // This should return the sky luminance (in cd/m^2) along the segment from 'camera' to the
///    // nearest atmosphere boundary in direction 'viewRay', as well as the transmittance along
///    // this segment.
///    vec3 GetSkyLuminance(vec3 camera, vec3 viewRay, vec3 sunDirection, out vec3 transmittance);
///
///    // This should return the sky luminance along the segment from 'camera' to 'p' (in cd/m^2),
///    // as well as the transmittance along this segment.
///    vec3 GetSkyLuminanceToPoint(vec3 camera, vec3 p, vec3 sunDirection, out vec3 transmittance);
///
///    // This should return the sun and sky illuminance (in lux) received on a surface patch
///    // located at 'p'. Writing the skyIlluminance is not strictly necessary but will result in a
///    // more realistic appearance during twilight when the sky is the only available light source.
///    vec3 GetSunAndSkyIlluminance(vec3 p, vec3 sunDirection, out vec3 skyIlluminance);
///
/// All coordinates are in planeto-centric cartesian coordinates in meters. There is also one
/// uniform available which could be of use:
///
///    // The sun-based illuminance at the given fragment position (in lux) if no atmosphere were
///    // present.
///    uniform float uSunIlluminance;
///
/// All other aspects of the atmospheres (like cloud rendering or water rendering) is handled
/// outside the model.
class ModelBase {
 public:
  /// Whenever the model parameters are changed (e.g. when the settings of CosmoScout VR got
  /// reloaded), this method will be called. The modelSettings parameter contains everything which
  /// the user passed to the corresponding "modelSettings" object in the settings.
  /// Use this method to perform any required pre-processing. You should also create and compile a
  /// fragment shader according to the description above. Later, this shader should be returned by
  /// the getShader() method.
  /// All computations are performed in a spherical atmosphere. The shader which calls the model API
  /// will ensure to transform the coordinates so that this also works for ellipsoidal planets.
  /// The method should return true if the shader needed to be recompiled.
  virtual bool init(
      nlohmann::json const& modelSettings, double planetRadius, double atmosphereRadius) = 0;

  /// Returns the fragment shader created during the last call to init(). See the class description
  /// above for more details.
  virtual GLuint getShader() const = 0;

  /// This will be called each time the atmosphere is drawn. Use this to set model-specific
  /// uniforms. Any required textures can be bound starting at the given texture unit. The method
  /// should return the next free texture unit. So if the atmospheric model does not require any
  /// textures, it should simply return startTextureUnit.
  virtual GLuint setUniforms(GLuint program, GLuint startTextureUnit) const = 0;
};

} // namespace csp::atmospheres

#endif // CSP_ATMOSPHERES_MODEL_BASE_HPP
