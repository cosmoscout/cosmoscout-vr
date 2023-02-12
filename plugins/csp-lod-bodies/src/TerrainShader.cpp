////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "TerrainShader.hpp"

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
  VistaShaderRegistry& reg = VistaShaderRegistry::GetInstance();

  // "Include" shared Uniforms / Functions
  cs::utils::replaceString(mVertexSource, "$VP_TERRAIN_SHADER_UNIFORMS",
      reg.RetrieveShader("VistaPlanetTerrainShaderUniforms.glsl"));
  cs::utils::replaceString(mVertexSource, "$VP_TERRAIN_SHADER_FUNCTIONS",
      reg.RetrieveShader("VistaPlanetTerrainShaderFunctions.vert"));

  // "Include" shared Uniforms / Functions
  cs::utils::replaceString(mTessEvalSource, "$VP_TERRAIN_SHADER_UNIFORMS",
      reg.RetrieveShader("VistaPlanetTerrainShaderUniforms.glsl"));
  cs::utils::replaceString(mTessEvalSource, "$VP_TERRAIN_SHADER_FUNCTIONS",
      reg.RetrieveShader("VistaPlanetTerrainShaderFunctions.vert"));

  // "Include" shared Uniforms / Functions
  cs::utils::replaceString(mFragmentSource, "$VP_TERRAIN_SHADER_UNIFORMS",
      reg.RetrieveShader("VistaPlanetTerrainShaderUniforms.glsl"));
  cs::utils::replaceString(mFragmentSource, "$VP_TERRAIN_SHADER_FUNCTIONS",
      reg.RetrieveShader("VistaPlanetTerrainShaderFunctions.frag"));

  mShader = VistaGLSLShader();
  mShader.InitVertexShaderFromString(mVertexSource);
  mShader.InitFragmentShaderFromString(mFragmentSource);
  mShader.InitShaderFromString(GL_TESS_CONTROL_SHADER, mTessContSource);
  mShader.InitShaderFromString(GL_TESS_EVALUATION_SHADER, mTessEvalSource);
  mShader.Link();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
