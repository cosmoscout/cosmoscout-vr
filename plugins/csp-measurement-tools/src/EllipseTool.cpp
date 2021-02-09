////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "EllipseTool.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-scene/CelestialAnchorNode.hpp"
#include "../../../src/cs-utils/convert.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

namespace csp::measurementtools {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* EllipseTool::SHADER_VERT = R"(
#version 330

layout(location=0) in vec3 iPosition;

out vec4 vPosition;

uniform mat4 uMatModelView;
uniform mat4 uMatProjection;

void main()
{
    vPosition   = uMatModelView * vec4(iPosition, 1.0);
    gl_Position = uMatProjection * vPosition;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* EllipseTool::SHADER_FRAG = R"(
#version 330

in vec4 vPosition;

uniform float uFarClip;
uniform vec3 uColor;

layout(location = 0) out vec4 oColor;

void main()
{
    oColor = vec4(uColor, 1.0);
   
    // linearize depth value
    gl_FragDepth = length(vPosition.xyz) / uFarClip;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

EllipseTool::EllipseTool(std::shared_ptr<cs::core::InputManager> const& pInputManager,
    std::shared_ptr<cs::core::SolarSystem> const&                       pSolarSystem,
    std::shared_ptr<cs::core::Settings> const&                          settings,
    std::shared_ptr<cs::core::TimeControl> const& pTimeControl, std::string const& sCenter,
    std::string const& sFrame)
    : mSolarSystem(pSolarSystem)
    , mSettings(settings)
    , mCenterHandle(pInputManager, pSolarSystem, settings, pTimeControl, sCenter, sFrame)
    , mAxes({glm::dvec3(pSolarSystem->getObserver().getAnchorScale(), 0.0, 0.0),
          glm::dvec3(0.0, pSolarSystem->getObserver().getAnchorScale(), 0.0)})
    , mHandles({std::make_unique<cs::core::tools::Mark>(
                    pInputManager, pSolarSystem, settings, pTimeControl, sCenter, sFrame),
          std::make_unique<cs::core::tools::Mark>(
              pInputManager, pSolarSystem, settings, pTimeControl, sCenter, sFrame)}) {

  mShader.InitVertexShaderFromString(SHADER_VERT);
  mShader.InitFragmentShaderFromString(SHADER_FRAG);
  mShader.Link();

  mUniforms.modelViewMatrix  = mShader.GetUniformLocation("uMatModelView");
  mUniforms.projectionMatrix = mShader.GetUniformLocation("uMatProjection");
  mUniforms.color            = mShader.GetUniformLocation("uColor");
  mUniforms.farClip          = mShader.GetUniformLocation("uFarClip");

  mVBO.Bind(GL_ARRAY_BUFFER);
  mVBO.BufferData(mNumSamples * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);
  mVBO.Release();

  mVAO.EnableAttributeArray(0);
  mVAO.SpecifyAttributeArrayFloat(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0, &mVBO);

  auto* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  mAnchor = std::make_shared<cs::scene::CelestialAnchorNode>(
      pSG->GetRoot(), pSG->GetNodeBridge(), "", sCenter, sFrame);
  mSolarSystem->registerAnchor(mAnchor);

  mOpenGLNode.reset(pSG->NewOpenGLNode(mAnchor.get(), this));

  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mOpenGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eOpaqueNonHDR));

  mCenterHandle.pLngLat.connect([this](glm::dvec2 const& /*lngLat*/) {
    auto center = mCenterHandle.getAnchor()->getAnchorPosition();
    auto body   = mSolarSystem->getBody(mCenterHandle.getAnchor()->getCenterName());
    auto radii  = body->getRadii();

    if (mFirstUpdate) {
      for (int i(0); i < 2; ++i) {
        glm::dvec2 lngLat2 = cs::utils::convert::cartesianToLngLat(center + mAxes.at(i), radii);

        mHandles.at(i)->pLngLat.setWithEmitForAllButOne(lngLat2, mHandleConnections.at(i));
      }
      mFirstUpdate = false;
    }

    mAxes.at(0) = mHandles.at(0)->getAnchor()->getAnchorPosition() - center;
    mAxes.at(1) = mHandles.at(1)->getAnchor()->getAnchorPosition() - center;

    mVerticesDirty = true;
  });

  for (int i(0); i < 2; ++i) {
    mHandleConnections.at(i) = mHandles.at(i)->pLngLat.connect([this, i](glm::dvec2 const& /*p*/) {
      auto center    = mCenterHandle.getAnchor()->getAnchorPosition();
      mAxes.at(i)    = mHandles.at(i)->getAnchor()->getAnchorPosition() - center;
      mVerticesDirty = true;
    });
  }

  // Whenever the height scale changes our vertex positions need to be updated.
  mScaleConnection =
      mSettings->mGraphics.pHeightScale.connect([this](float /*h*/) { mVerticesDirty = true; });

  // Delete the tool when the center handle is deleted.
  pShouldDelete.connectFrom(mCenterHandle.pShouldDelete);

  // Adjust the color of all handles according to the tool's color.
  mCenterHandle.pColor.connectFrom(pColor);
  mHandles[0]->pColor.connectFrom(pColor);
  mHandles[1]->pColor.connectFrom(pColor);

  // Adjust the scaling of all handles according to the center handle's scaling.
  mHandles[0]->pScaleDistance.connectFrom(mCenterHandle.pScaleDistance);
  mHandles[1]->pScaleDistance.connectFrom(mCenterHandle.pScaleDistance);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EllipseTool::~EllipseTool() {
  // disconnect slots
  mSettings->mGraphics.pHeightScale.disconnect(mScaleConnection);

  mSolarSystem->unregisterAnchor(mAnchor);
  pShouldDelete.disconnect();
  pColor.disconnectAll();

  auto* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  pSG->GetRoot()->DisconnectChild(mOpenGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void EllipseTool::setCenterName(std::string const& name) {
  mCenterHandle.getAnchor()->setCenterName(name);
  mHandles.at(0)->getAnchor()->setCenterName(name);
  mHandles.at(1)->getAnchor()->setCenterName(name);
  mAnchor->setCenterName(name);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& EllipseTool::getCenterName() const {
  return mAnchor->getCenterName();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void EllipseTool::setFrameName(std::string const& name) {
  mCenterHandle.getAnchor()->setFrameName(name);
  mHandles.at(0)->getAnchor()->setFrameName(name);
  mHandles.at(1)->getAnchor()->setFrameName(name);
  mAnchor->setFrameName(name);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& EllipseTool::getFrameName() const {
  return mAnchor->getFrameName();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

FlagTool const& EllipseTool::getCenterHandle() const {
  return mCenterHandle;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cs::core::tools::Mark const& EllipseTool::getFirstHandle() const {
  return *mHandles.at(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cs::core::tools::Mark const& EllipseTool::getSecondHandle() const {
  return *mHandles.at(1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

FlagTool& EllipseTool::getCenterHandle() {
  return mCenterHandle;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cs::core::tools::Mark& EllipseTool::getFirstHandle() {
  return *mHandles.at(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

cs::core::tools::Mark& EllipseTool::getSecondHandle() {
  return *mHandles.at(1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void EllipseTool::setNumSamples(int const& numSamples) {
  mNumSamples    = numSamples;
  mVerticesDirty = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void EllipseTool::calculateVertices() {
  auto body   = mSolarSystem->getBody(mCenterHandle.getAnchor()->getCenterName());
  auto radii  = body->getRadii();
  auto center = mCenterHandle.getAnchor()->getAnchorPosition();
  auto normal = cs::utils::convert::lngLatToNormal(mCenterHandle.pLngLat.get());

  mAnchor->setAnchorPosition(center);

  glm::dvec3 north(0, 1, 0);
  glm::dvec3 east = glm::normalize(glm::cross(north, normal));
  north           = glm::normalize(glm::cross(normal, east));

  std::vector<glm::vec3> vRelativePositions(mNumSamples);
  for (int i = 0; i < mNumSamples; ++i) {
    double phi = glm::mix(0.0, 2.0 * glm::pi<double>(), 1.0 * i / (mNumSamples - 1));
    double x   = std::sin(phi);
    double y   = std::cos(phi);

    glm::dvec3 absPosition = center + x * mAxes[0] + y * mAxes[1];
    glm::dvec2 lngLat      = cs::utils::convert::cartesianToLngLat(absPosition, radii);

    double height =
        mSolarSystem->getBody(mCenterHandle.getAnchor()->getCenterName())->getHeight(lngLat);
    height *= mSettings->mGraphics.pHeightScale.get();
    absPosition = cs::utils::convert::toCartesian(lngLat, radii, height);

    vRelativePositions[i] = absPosition - center;
  }

  mVBO.Bind(GL_ARRAY_BUFFER);
  mVBO.BufferSubData(0, vRelativePositions.size() * sizeof(glm::vec3), vRelativePositions.data());
  mVBO.Release();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void EllipseTool::update() {
  mCenterHandle.update();
  mHandles.at(0)->update();
  mHandles.at(1)->update();

  if (mVerticesDirty) {
    calculateVertices();
    mVerticesDirty = false;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool EllipseTool::Do() {
  glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_LINE_BIT);

  // enable alpha blending
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // enable and configure line rendering
  glEnable(GL_LINE_SMOOTH);
  glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
  glLineWidth(5);

  std::array<GLfloat, 16> glMatMV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());

  mShader.Bind();
  mVAO.Bind();
  glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glMatMV.data());
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());

  mShader.SetUniform(mUniforms.color, pColor.get().r, pColor.get().g, pColor.get().b);
  mShader.SetUniform(mUniforms.farClip, cs::utils::getCurrentFarClipDistance());

  // draw the linestrip
  glDrawArrays(GL_LINE_STRIP, 0, mNumSamples);
  mVAO.Release();
  mShader.Release();

  glPopAttrib();
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool EllipseTool::GetBoundingBox(VistaBoundingBox& bb) {
  std::array fMin{-0.1F, -0.1F, -0.1F};
  std::array fMax{0.1F, 0.1F, 0.1F};

  bb.SetBounds(fMin.data(), fMax.data());
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::measurementtools
