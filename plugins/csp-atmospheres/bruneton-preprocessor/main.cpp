////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include <GL/glew.h>
#include <SDL2/SDL.h>
#include <fstream>

#include "../../../../src/cs-utils/filesystem.hpp"

#include "Params.hpp"
#include "Preprocessor.hpp"
#include "csv.hpp"

// -------------------------------------------------------------------------------------------------

void printHelp() {
  std::cout << "Welcome to the Atmosphere Preprocessor! Usage:" << std::endl;
  std::cout << std::endl;
  std::cout << "  ./bruneton-preprocessor <input JSON> <output directory>" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// This preprocessor loads the CSV files containing the scattering data and precomputes the       //
// textures which are needed to render the atmosphere. The preprocessor is based on the           //
// implementation by Eric Bruneton: https://github.com/ebruneton/precomputed_atmospheric_scattering
// See the Preprocessor class for more information.                                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {

  if (argc <= 2) {
    printHelp();
    return 0;
  }

  std::string cInput(argv[1]);
  std::string cOutput(argv[2]);

  // Try parsing the atmosphere settings.
  std::ifstream stream(cInput, std::ios::in);

  // First test if input file exists.
  if (!stream.is_open()) {
    std::cerr << "Failed to open input file: " << cInput << std::endl;
    return 1;
  }

  // Then try to parse the JSON.
  nlohmann::json json;
  stream >> json;

  Params params;

  // And convert it to the Params struct.
  try {
    params = json;
  } catch (std::exception const& e) {
    std::cerr << "Failed to parse atmosphere parameters: " << e.what() << std::endl;
    return 1;
  }

  // Read all the CSV files.
  // clang-format off
  uint32_t densityCount = 0;

  params.mMolecules.mPhase      = csv::readPhase(params.mMolecules.mPhaseFile, params.mWavelengths);
  params.mMolecules.mDensity    = csv::readDensity(params.mMolecules.mDensityFile, densityCount);
  params.mMolecules.mScattering = csv::readExtinction(params.mMolecules.mBetaScaFile, params.mWavelengths);
  params.mMolecules.mAbsorption = csv::readExtinction(params.mMolecules.mBetaAbsFile, params.mWavelengths);

  params.mAerosols.mPhase      = csv::readPhase(params.mAerosols.mPhaseFile, params.mWavelengths);
  params.mAerosols.mDensity    = csv::readDensity(params.mAerosols.mDensityFile, densityCount);
  params.mAerosols.mScattering = csv::readExtinction(params.mAerosols.mBetaScaFile, params.mWavelengths);
  params.mAerosols.mAbsorption = csv::readExtinction(params.mAerosols.mBetaAbsFile, params.mWavelengths);

  if (params.mOzone) {
    params.mOzone->mDensity    = csv::readDensity(params.mOzone->mDensityFile, densityCount);
    params.mOzone->mAbsorption = csv::readExtinction(params.mOzone->mBetaAbsFile, params.mWavelengths);
  } else {
    params.mOzone = Params::AbsorbingComponent();
    params.mOzone->mDensity    = std::vector<float>(densityCount, 0.0);
    params.mOzone->mAbsorption = std::vector<float>(params.mWavelengths.size(), 0.0);
  }

  if (params.mRefractiveIndexFile) {
    params.mRefractiveIndex = csv::readIoR(*params.mRefractiveIndexFile, params.mWavelengths);
  } else {
    params.mRefractiveIndex = std::vector<float>(params.mWavelengths.size(), 0.0);
  }
  // clang-format on

  // Check for valid wavelengths.
  if (params.mWavelengths.size() < 3) {
    throw std::runtime_error(
        "At least three different wavelengths should be given in the scattering data!");
  } else if (params.mWavelengths.size() == 3 &&
             (params.mWavelengths[0] != Preprocessor::kLambdaB ||
                 params.mWavelengths[1] != Preprocessor::kLambdaG ||
                 params.mWavelengths[2] != Preprocessor::kLambdaR)) {
    throw std::runtime_error("If three different wavelengths are given in the scattering data, "
                             "they should be exactly for 440 nm, 550 nm, and 680 nm!");
  }

  // Initialize SDL.
  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    SDL_Log("Unable to initialize SDL: %s", SDL_GetError());
    return 1;
  }

  // Set OpenGL version (3.3)
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

  // Create a window (invisible).
  SDL_Window* window = SDL_CreateWindow("OpenGL", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
      0, // width
      0, // height
      SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);

  if (!window) {
    SDL_Log("Could not create window: %s", SDL_GetError());
    SDL_Quit();
    return 1;
  }

  // Create OpenGL context.
  SDL_GLContext context = SDL_GL_CreateContext(window);

  // Initialize GLEW.
  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    SDL_Log("Failed to initialize GLEW");
    SDL_GL_DeleteContext(context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 1;
  }

  // Finally, run the preprocessor.
  Preprocessor preprocessor(params);
  preprocessor.run(params.mMultiScatteringOrder.get() + 1);

  // Create the output directory if it does not exist.
  cs::utils::filesystem::createDirectoryRecursively(boost::filesystem::system_complete(cOutput));

  // Save the precomputed textures.
  preprocessor.save(cOutput);

  // Clean up.
  SDL_GL_DeleteContext(context);
  SDL_DestroyWindow(window);
  SDL_Quit();

  return 0;
}
