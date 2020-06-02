////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

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

  cs::utils::replaceString(mVertexSource, "$VP_TERRAIN_SHADER_FUNCTIONS",
      reg.RetrieveShader("VistaPlanetTerrainShaderFunctions.vert"));
  cs::utils::replaceString(mVertexSource, "$VP_TERRAIN_SHADER_UNIFORMS",
      reg.RetrieveShader("VistaPlanetTerrainShaderUniforms.glsl"));

  cs::utils::replaceString(mFragmentSource, "$VP_TERRAIN_SHADER_FUNCTIONS",
      reg.RetrieveShader("VistaPlanetTerrainShaderFunctions.frag"));
  cs::utils::replaceString(mFragmentSource, "$VP_TERRAIN_SHADER_UNIFORMS",
      reg.RetrieveShader("VistaPlanetTerrainShaderUniforms.glsl"));

  mShader = VistaGLSLShader();
  mShader.InitVertexShaderFromString(mVertexSource);
  mShader.InitFragmentShaderFromString(mFragmentSource);
  mShader.Link();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
