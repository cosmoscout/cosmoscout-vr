////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "SunFlare.hpp"

#include "../../../src/cs-core/Settings.hpp"
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

const char* SunFlare::QUAD_VERT = R"(
#version 330

out vec2 vTexCoords;
out float fDepth;

uniform float uAspect;
uniform mat4 uMatModelView;
uniform mat4 uMatProjection;

void main()
{
    vec4 posVS = uMatModelView * vec4(0, 0, 0, 1);
    fDepth = length(posVS.xyz);

    vec4 posP = uMatProjection * posVS;
    float scale = length(uMatModelView[0]) / length(posVS.xyz);

    if (posP.w < 0) {
        gl_Position = vec4(0);
        return;
    }

    posP /= posP.w;

    float h = scale * 10e10;
    float w = h / uAspect;

    posP.z = 0.999;

    switch (gl_VertexID) {
        case 0:
            posP.xy += vec2(-w,  h);
            vTexCoords = vec2(-1, 1);
            break;
        case 1:
            posP.xy += vec2( w,  h);
            vTexCoords = vec2(1, 1);
            break;
        case 2:
            posP.xy += vec2(-w, -h);
            vTexCoords = vec2(-1, -1);
            break;
        default:
            posP.xy += vec2( w, -h);
            vTexCoords = vec2(1, -1);
            break;
    }

    gl_Position = posP;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* SunFlare::QUAD_FRAG = R"(
#version 330

uniform vec3 uCcolor;
uniform float uFarClip;

in vec2 vTexCoords;
in float fDepth;

layout(location = 0) out vec3 oColor;

void main()
{
    // sun disc
    float dist = length(vTexCoords) * 100;
    float glow = exp(1.0 - dist);
    oColor = uCcolor * glow;
    
    // sun glow
    dist = min(1.0, length(vTexCoords));
    glow = 1.0 - pow(dist, 0.05);
    oColor += uCcolor * glow * 2;

    gl_FragDepth = fDepth / uFarClip;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

SunFlare::SunFlare(std::shared_ptr<cs::core::Settings> settings,
    std::shared_ptr<Plugin::Settings> pluginSettings, std::string const& sCenterName,
    std::string const& sFrameName, double tStartExistence, double tEndExistence)
    : cs::scene::CelestialObject(sCenterName, sFrameName, tStartExistence, tEndExistence)
    , mSettings(std::move(settings))
    , mPluginSettings(std::move(pluginSettings)) {
  mShader.InitVertexShaderFromString(QUAD_VERT);
  mShader.InitFragmentShaderFromString(QUAD_FRAG);
  mShader.Link();

  // Add to scenegraph.
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eAtmospheres) + 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SunFlare::~SunFlare() {
  VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SunFlare::Do() {
  if (mPluginSettings->mEnableSunFlares.get() && getIsInExistence() &&
      !mSettings->mGraphics.pEnableHDR.get()) {
    cs::utils::FrameTimings::ScopedTimer timer("SunFlare");
    // get viewport to draw dot with correct aspect ration
    std::array<GLint, 4> viewport{};
    glGetIntegerv(GL_VIEWPORT, viewport.data());
    float fAspect = 1.F * viewport.at(2) / viewport.at(3);

    // get modelview and projection matrices
    std::array<GLfloat, 16> glMatMV{};
    std::array<GLfloat, 16> glMatP{};
    glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
    glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
    auto matMV = glm::make_mat4x4(glMatMV.data()) * glm::mat4(getWorldTransform());

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glDepthMask(GL_FALSE);

    // draw simple dot
    mShader.Bind();
    glUniformMatrix4fv(
        mShader.GetUniformLocation("uMatModelView"), 1, GL_FALSE, glm::value_ptr(matMV));
    glUniformMatrix4fv(mShader.GetUniformLocation("uMatProjection"), 1, GL_FALSE, glMatP.data());
    mShader.SetUniform(
        mShader.GetUniformLocation("uCcolor"), pColor.get()[0], pColor.get()[1], pColor.get()[2]);
    mShader.SetUniform(mShader.GetUniformLocation("uAspect"), fAspect);
    mShader.SetUniform(
        mShader.GetUniformLocation("uFarClip"), cs::utils::getCurrentFarClipDistance());

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    mShader.Release();

    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SunFlare::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::trajectories
