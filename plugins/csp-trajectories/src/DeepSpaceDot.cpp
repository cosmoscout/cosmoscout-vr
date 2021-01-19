////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DeepSpaceDot.hpp"

#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-utils/FrameTimings.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include <glm/gtc/type_ptr.hpp>
#include <utility>

namespace csp::trajectories {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* DeepSpaceDot::QUAD_VERT = R"(
#version 330

out vec2 vTexCoords;
out float fDepth;

uniform float uAspect;
uniform mat4 uMatModelView;
uniform mat4 uMatProjection;

void main()
{
    vec4 pos = uMatModelView * vec4(0, 0, 0, 1);
    fDepth = length(pos.xyz);

    pos = uMatProjection * pos;

    if (pos.w < 0) {
        gl_Position = vec4(0);
        return;
    }

    pos /= pos.w;

    float h = 0.0075;
    float w = h / uAspect;

    pos.z = 0.9999999;

    switch (gl_VertexID) {
        case 0:
            pos.xy += vec2(-w,  h);
            vTexCoords = vec2(-1, 1);
            break;
        case 1:
            pos.xy += vec2( w,  h);
            vTexCoords = vec2(1, 1);
            break;
        case 2:
            pos.xy += vec2(-w, -h);
            vTexCoords = vec2(-1, -1);
            break;
        default:
            pos.xy += vec2( w, -h);
            vTexCoords = vec2(1, -1);
            break;
    }

    gl_Position = pos;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* DeepSpaceDot::QUAD_FRAG = R"(
#version 330

uniform vec3 uColor;
uniform float uFarClip;

in vec2 vTexCoords;
in float fDepth;

layout(location = 0) out vec4 oColor;

void main()
{
    float dist = length(vTexCoords);
    float blob = pow(dist, 10.0);
    oColor  = mix(vec4(uColor, 1.0), vec4(0), blob);
    
    // substract a small value to prevent depth fighting with trajectories
    gl_FragDepth = fDepth / uFarClip - 0.00001;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

DeepSpaceDot::DeepSpaceDot(std::shared_ptr<Plugin::Settings> pluginSettings,
    std::shared_ptr<cs::core::Settings> const& settings, std::string const& anchorName)
    : mPluginSettings(std::move(pluginSettings)) {

  settings->initAnchor(*this, anchorName);
  setRadii(glm::dvec3(0.0));

  mShader.InitVertexShaderFromString(QUAD_VERT);
  mShader.InitFragmentShaderFromString(QUAD_FRAG);
  mShader.Link();

  mUniforms.modelViewMatrix  = mShader.GetUniformLocation("uMatModelView");
  mUniforms.projectionMatrix = mShader.GetUniformLocation("uMatProjection");
  mUniforms.color            = mShader.GetUniformLocation("uColor");
  mUniforms.aspect           = mShader.GetUniformLocation("uAspect");
  mUniforms.farClip          = mShader.GetUniformLocation("uFarClip");

  // Add to scenegraph.
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems) - 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DeepSpaceDot::~DeepSpaceDot() {
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DeepSpaceDot::Do() {
  if (mPluginSettings->mEnablePlanetMarks.get() && getIsInExistence() && pVisible.get()) {
    cs::utils::FrameTimings::ScopedTimer timer("Planet Marks");
    // get viewport to draw dot with correct aspect ration
    std::array<GLint, 4> viewport{};
    glGetIntegerv(GL_VIEWPORT, viewport.data());
    float fAspect = 1.F * viewport.at(2) / viewport.at(3);

    // get model view and projection matrices
    std::array<GLfloat, 16> glMatMV{};
    std::array<GLfloat, 16> glMatP{};
    glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
    glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
    auto matMV = glm::make_mat4x4(glMatMV.data()) * glm::mat4(getWorldTransform());

    glEnable(GL_BLEND);
    glDepthMask(GL_FALSE);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // draw simple dot
    mShader.Bind();
    glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glm::value_ptr(matMV));
    glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());
    mShader.SetUniform(mUniforms.color, pColor.get()[0], pColor.get()[1], pColor.get()[2]);
    mShader.SetUniform(mUniforms.aspect, fAspect);
    mShader.SetUniform(mUniforms.farClip, cs::utils::getCurrentFarClipDistance());
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    mShader.Release();

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DeepSpaceDot::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::trajectories
