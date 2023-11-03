////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "TerrainShader.hpp"

#include "../../../src/cs-utils/filesystem.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaShaderRegistry.h>

#include <utility>

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

TerrainShader::TerrainShader(std::string vertexSource, std::string fragmentSource)
    : mVertexSource(std::move(vertexSource))
    , mFragmentSource(std::move(fragmentSource)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TerrainShader::bind() {
  if (mShaderDirty) {
    compile();
    mShaderDirty = false;
  }

  mShader.Bind();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TerrainShader::release() {
  mShader.Release();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TerrainShader::compile() {
  cs::utils::replaceString(mVertexSource, "$VP_TERRAIN_SHADER_FUNCTIONS",
      cs::utils::filesystem::loadToString(
          "../share/resources/shaders/VistaPlanetTerrainShaderFunctions.vert"));
  cs::utils::replaceString(mVertexSource, "$VP_TERRAIN_SHADER_UNIFORMS",
      cs::utils::filesystem::loadToString(
          "../share/resources/shaders/VistaPlanetTerrainShaderUniforms.glsl"));

  cs::utils::replaceString(mFragmentSource, "$VP_TERRAIN_SHADER_FUNCTIONS",
      cs::utils::filesystem::loadToString(
          "../share/resources/shaders/VistaPlanetTerrainShaderFunctions.frag"));
  cs::utils::replaceString(mFragmentSource, "$VP_TERRAIN_SHADER_UNIFORMS",
      cs::utils::filesystem::loadToString(
          "../share/resources/shaders/VistaPlanetTerrainShaderUniforms.glsl"));

  mShader = VistaGLSLShader();
  mShader.InitVertexShaderFromString(mVertexSource);
  mShader.InitFragmentShaderFromString(mFragmentSource);
  mShader.Link();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
