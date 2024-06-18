////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: 2017 Eric Bruneton
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PREPROCESSOR_HPP
#define PREPROCESSOR_HPP

#include "Metadata.hpp"
#include "Params.hpp"

#include <GL/glew.h>
#include <array>
#include <functional>
#include <glm/glm.hpp>
#include <string>
#include <vector>

/// The preprocessor is based on this class by Eric Bruneton:
/// https://github.com/ebruneton/precomputed_atmospheric_scattering/blob/master/atmosphere/model.h
/// While we refactored / restyled large parts of the code, the overall flow of control remains the
/// same.
/// See the source file for more information.
class Preprocessor {
 public:
  /// If only three wavelengths are used during preprocessing, these three are used:
  static constexpr float kLambdaR = 680.0;
  static constexpr float kLambdaG = 550.0;
  static constexpr float kLambdaB = 440.0;

  /// The constructor of the class takes all parameters which define the attributes of the
  /// atmosphere. It will allocate various GPU resources.
  Preprocessor(Params params);
  ~Preprocessor();

  /// This will preprocess the multiple scattering up to the given number. Setting this to one will
  /// disable multiple scattering.
  void run(unsigned int numScatteringOrders);

  /// This will save the precomputed textures to the given directory.
  void save(std::string const& directory);

 private:
  void precompute(GLuint fbo, GLuint deltaIrradianceTexture, GLuint deltaMoleculesScatteringTexture,
      GLuint deltaAerosolsScatteringTexture, GLuint deltaScatteringDensityTexture,
      GLuint deltaMultipleScatteringTexture, glm::vec3 const& lambdas,
      glm::mat3 const& luminanceFromRadiance, bool blend, unsigned int numScatteringOrders);

  void updatePhaseFunctionTexture(
      std::vector<Params::ScatteringComponent> const& scatteringComponents,
      glm::vec3 const&                                lambdas);

  const Params  mParams;
  const int32_t mScatteringTextureWidth;
  const int32_t mScatteringTextureHeight;
  const int32_t mScatteringTextureDepth;

  Metadata mMetadata;

  std::function<std::string(glm::vec3 const&)> mGlslHeaderFactory;

  // To optimize resource usage, this texture stores single molecule-scattering plus all
  // multiple-scattering contributions. The single aerosols scattering is stored in an extra
  // texture.
  GLuint mMultipleScatteringTexture       = 0;
  GLuint mSingleAerosolsScatteringTexture = 0;

  GLuint mPhaseTexture         = 0;
  GLuint mDensityTexture       = 0;
  GLuint mTransmittanceTexture = 0;
  GLuint mMuDeviationTexture   = 0;
  GLuint mIrradianceTexture    = 0;
  GLuint mFullScreenQuadVAO    = 0;
  GLuint mFullScreenQuadVBO    = 0;
};

#endif // PREPROCESSOR_HPP
