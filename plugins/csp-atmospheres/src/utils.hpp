////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ATMOSPHERES_UTILS_HPP
#define CSP_ATMOSPHERES_UTILS_HPP

#include <GL/glew.h>
#include <string>
#include <tuple>

#include <glm/glm.hpp>
#include <VistaBase/VistaStreamUtils.h>

#include <stb_image.h>

namespace csp::atmospheres::utils {

struct Uniforms {
  uint32_t sunDir                    = 0;
  uint32_t sunInfo                   = 0;
  uint32_t time                      = 0;
  uint32_t depthBuffer               = 0;
  uint32_t colorBuffer               = 0;
  uint32_t waterLevel                = 0;
  uint32_t cloudTexture              = 0;
  uint32_t cloudTypeTexture          = 0;
  uint32_t noiseTexture2D            = 0;
  uint32_t cloudAltitude             = 0;
  uint32_t limbLuminanceTexture      = 0;
  uint32_t inverseModelViewMatrix    = 0;
  uint32_t inverseProjectionMatrix   = 0;
  uint32_t scaleMatrix               = 0;
  uint32_t modelMatrix               = 0;
  uint32_t modelViewProjectionMatrix = 0;
  uint32_t shadowCoordinates         = 0;
  uint32_t noiseTexture              = 0;
  uint32_t cloudDensityMultiplier    = 0;
  uint32_t cloudAbsorption           = 0;
  uint32_t coverageExponent          = 0;
  uint32_t cloudCutoff               = 0;
  uint32_t cloudLFRepetitionScale    = 0;
  uint32_t cloudHFRepetitionScale    = 0;

  uint32_t cloudQuality              = 0;
  uint32_t cloudMaxSamples           = 0;
  uint32_t cloudJitter               = 0;
  uint32_t cloudTypeExponent         = 0;
  uint32_t cloudRangeMin             = 0;
  uint32_t cloudRangeMax             = 0;
  uint32_t cloudTypeMin              = 0;
  uint32_t cloudTypeMax              = 0;
  
  uint32_t cloudInterpolationStrideScale = 0;

  // Only used by the panorama shader.
  uint32_t atmoPanoUniforms = 0;

  // Only used by the skydome shader.
  uint32_t sunElevation = 0;
};

// Loads a 2D tiff texture containing precomputed values for the atmosphere. This is for instance
// used for the transmittance texture. It returns a tuple containing the OpenGL texture handle and
// size of the texture.
std::tuple<GLuint, glm::ivec2> read2DTexture(std::string const& path);

// Loads a 3D tiff texture containing precomputed values for the atmosphere. This is for instance
// used for the single scattering texture. It returns a tuple containing the OpenGL texture handle
// and size of the texture.
std::tuple<GLuint, glm::ivec3> read3DTexture(std::string const& path);

std::vector<float> readTexture(std::string const& path, int *width, int *height, int *channels);
} // namespace csp::atmospheres::utils

#endif // CSP_ATMOSPHERES_UTILS_HPP
