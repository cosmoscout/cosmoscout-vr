////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DipStrikeTool.hpp"

#include "logger.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/InputManager.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"
#include "../../../src/cs-core/tools/DeletableMark.hpp"
#include "../../../src/cs-scene/CelestialAnchorNode.hpp"
#include "../../../src/cs-utils/convert.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace csp::measurementtools {

////////////////////////////////////////////////////////////////////////////////////////////////////

const int DipStrikeTool::RESOLUTION = 100;

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* DipStrikeTool::SHADER_VERT = R"(
#version 330

layout(location=0) in vec2 iPosition;

out vec4 vPosition;
out vec2 vTexcoord;

uniform mat4 uMatModelView;
uniform mat4 uMatProjection;

void main()
{
    vPosition   = uMatModelView * vec4(iPosition.x, 0, iPosition.y, 1.0);
    vTexcoord   = iPosition;
    gl_Position = uMatProjection * vPosition;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* DipStrikeTool::SHADER_FRAG = R"(
#version 330

in vec4 vPosition;
in vec2 vTexcoord;

uniform float uOpacity;
uniform float uFarClip;

layout(location = 0) out vec4 oColor;

void main()
{
    if (uOpacity == 0)
        discard;

    float lines = 10;
    float spacing = 1.0/lines;

    vec2 linesMod = mod(vec2(1.0) - vTexcoord, vec2(spacing)); 

    float dipWidth    = fwidth(vTexcoord.y) * linesMod.x * 100;
    float strikeWidth = fwidth(vTexcoord.x) * 2;

    linesMod.x = (linesMod.x > 0.5 * spacing) ? spacing - linesMod.x : linesMod.x;
    linesMod.y = (linesMod.y > 0.5 * spacing) ? spacing - linesMod.y : linesMod.y;

    float dipAlpha    = 1.0 - clamp(abs(linesMod.y / dipWidth), 0, 1);
    float strikeAlpha = 1.0 - clamp(abs(linesMod.x / strikeWidth), 0, 1);

    oColor = vec4(0.5, 0.7, 1.0, uOpacity);
    oColor.rgb = mix(vec3(1), oColor.rgb, 1.0 - 0.4 * strikeAlpha);
    oColor.rgb = mix(vec3(1), oColor.rgb, 1.0 - 0.7 * dipAlpha);

    gl_FragDepth = length(vPosition.xyz) / uFarClip;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

DipStrikeTool::DipStrikeTool(std::shared_ptr<cs::core::InputManager> const& pInputManager,
    std::shared_ptr<cs::core::SolarSystem> const&                           pSolarSystem,
    std::shared_ptr<cs::core::Settings> const&                              settings,
    std::shared_ptr<cs::core::TimeControl> const& pTimeControl, std::string const& sCenter,
    std::string const& sFrame)
    : MultiPointTool(pInputManager, pSolarSystem, settings, pTimeControl, sCenter, sFrame)
    , mGuiArea(std::make_unique<cs::gui::WorldSpaceGuiArea>(600, 260))
    , mGuiItem(std::make_unique<cs::gui::GuiItem>(
          "file://{toolZoom}../share/resources/gui/dipstrike.html")) {

  // create the shader
  mShader.InitVertexShaderFromString(SHADER_VERT);
  mShader.InitFragmentShaderFromString(SHADER_FRAG);
  mShader.Link();

  mUniforms.modelViewMatrix  = mShader.GetUniformLocation("uMatModelView");
  mUniforms.projectionMatrix = mShader.GetUniformLocation("uMatProjection");
  mUniforms.opacity          = mShader.GetUniformLocation("uOpacity");
  mUniforms.farClip          = mShader.GetUniformLocation("uFarClip");

  auto* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  // create a a CelestialAnchorNode for the larger circular plane
  // it will be moved to the centroid of all points when a point is moved
  mPlaneAnchor = std::make_shared<cs::scene::CelestialAnchorNode>(
      pSG->GetRoot(), pSG->GetNodeBridge(), "", sCenter, sFrame);
  mSolarSystem->registerAnchor(mPlaneAnchor);

  // attach this as OpenGLNode to mPlaneAnchor
  mPlaneOpenGLNode.reset(pSG->NewOpenGLNode(mPlaneAnchor.get(), this));

  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mPlaneOpenGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems));

  // create a a CelestialAnchorNode for the user interface
  // it will be moved to the center of all points when a point is moved
  // and rotated in such a way, that it always faces the observer
  mGuiAnchor = std::make_shared<cs::scene::CelestialAnchorNode>(
      pSG->GetRoot(), pSG->GetNodeBridge(), "", sCenter, sFrame);
  mGuiAnchor->setAnchorScale(mSolarSystem->getObserver().getAnchorScale());
  mSolarSystem->registerAnchor(mGuiAnchor);

  // create the user interface
  mGuiTransform.reset(pSG->NewTransformNode(mGuiAnchor.get()));
  mGuiTransform->Translate(0.F, 0.9F, 0.F);
  mGuiTransform->Scale(0.0005F * static_cast<float>(mGuiArea->getWidth()),
      0.0005F * static_cast<float>(mGuiArea->getHeight()), 1.F);
  mGuiTransform->Rotate(VistaAxisAndAngle(VistaVector3D(0.F, 1.F, 0.F), -glm::pi<float>() / 2.F));
  mGuiArea->addItem(mGuiItem.get());
  mGuiArea->setUseLinearDepthBuffer(true);
  mGuiOpenGLNode.reset(pSG->NewOpenGLNode(mGuiTransform.get(), mGuiArea.get()));

  mInputManager->registerSelectable(mGuiOpenGLNode.get());

  mGuiItem->setCanScroll(false);
  mGuiItem->waitForFinishedLoading();

  // We use a zoom factor of 2.0 in order to increae the DPI of our world space UIs.
  mGuiItem->setZoomFactor(2.0);

  mGuiItem->registerCallback("deleteMe", "Call this to delete the tool.",
      std::function([this]() { pShouldDelete = true; }));

  mGuiItem->registerCallback("setAddPointMode", "Call this to enable creation of new points.",
      std::function([this](bool enable) {
        addPoint();
        pAddPointMode = enable;
      }));

  mGuiItem->registerCallback("setSize", "Sets the size of the dip and strike plane.",
      std::function([this](double val) { pSize = static_cast<float>(val); }));
  pSize.connectAndTouch([this](float value) {
    mGuiItem->callJavascript("CosmoScout.gui.setSliderValue", "setSize", false, value);
  });

  mGuiItem->registerCallback("setOpacity", "Sets the opacity of the dip and strike plane.",
      std::function([this](double val) { pOpacity = static_cast<float>(val); }));
  pOpacity.connectAndTouch([this](float value) {
    mGuiItem->callJavascript("CosmoScout.gui.setSliderValue", "setOpacity", false, value);
  });

  mGuiItem->setCursorChangeCallback([](cs::gui::Cursor c) { cs::core::GuiManager::setCursor(c); });

  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGuiOpenGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems));

  // update on height scale change
  mScaleConnection = mSettings->mGraphics.pHeightScale.connectAndTouch(
      [this](float /*h*/) { mVerticesDirty = true; });

  // create circle geometry
  std::vector<glm::vec2> vPositions;
  vPositions.reserve(RESOLUTION + 1);
  vPositions.emplace_back(glm::vec2(0, 0));
  for (int i(0); i < RESOLUTION; ++i) {
    float fFac(2.F * glm::pi<float>() * static_cast<float>(i) / (RESOLUTION - 1.F));
    vPositions.emplace_back(glm::vec2(std::cos(fFac), std::sin(fFac)));
  }

  mVBO.Bind(GL_ARRAY_BUFFER);
  mVBO.BufferData(vPositions.size() * sizeof(glm::vec2), vPositions.data(), GL_STATIC_DRAW);
  mVBO.Release();

  mVAO.EnableAttributeArray(0);
  mVAO.SpecifyAttributeArrayFloat(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), 0, &mVBO);

  // Update text.
  mTextConnection = pText.connectAndTouch(
      [this](std::string const& value) { mGuiItem->callJavascript("setText", value); });

  mGuiItem->registerCallback("onSetText",
      "This is called whenever the text input of the tool's name changes.",
      std::function(
          [this](std::string&& value) { pText.setWithEmitForAllButOne(value, mTextConnection); }));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

DipStrikeTool::~DipStrikeTool() {
  mSettings->mGraphics.pHeightScale.disconnect(mScaleConnection);
  mGuiItem->unregisterCallback("deleteMe");
  mGuiItem->unregisterCallback("setAddPointMode");
  mGuiItem->unregisterCallback("setSize");
  mGuiItem->unregisterCallback("setOpacity");
  mGuiItem->unregisterCallback("onSetText");

  mInputManager->unregisterSelectable(mGuiOpenGLNode.get());

  mSolarSystem->unregisterAnchor(mGuiAnchor);
  mSolarSystem->unregisterAnchor(mPlaneAnchor);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DipStrikeTool::setCenterName(std::string const& name) {
  cs::core::tools::MultiPointTool::setCenterName(name);
  mGuiAnchor->setCenterName(name);
  mPlaneAnchor->setCenterName(name);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DipStrikeTool::setFrameName(std::string const& name) {
  cs::core::tools::MultiPointTool::setFrameName(name);
  mGuiAnchor->setFrameName(name);
  mPlaneAnchor->setFrameName(name);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DipStrikeTool::onPointMoved() {
  mVerticesDirty = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DipStrikeTool::onPointAdded() {
  mVerticesDirty = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DipStrikeTool::onPointRemoved(int /*index*/) {
  mVerticesDirty = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DipStrikeTool::calculateDipAndStrike() {
  if (mPoints.empty()) {
    return;
  }

  auto body  = mSolarSystem->getBody(getCenterName());
  auto radii = body->getRadii();

  glm::dvec3 averagePosition{0};
  for (auto const& mark : mPoints) {
    averagePosition += mark->getAnchor()->getAnchorPosition() / static_cast<double>(mPoints.size());
  }

  // corrected average position (works for every height scale)
  // average position of the coordinates without height exaggeration
  glm::dvec3 averagePositionNorm(0.0);
  for (auto const& mark : mPoints) {
    // LongLat coordinate
    glm::dvec2 l =
        cs::utils::convert::cartesianToLngLat(mark->getAnchor()->getAnchorPosition(), radii);
    // Height of the point
    double h = body ? body->getHeight(l) : 0.0;
    // Cartesian coordinate with height
    glm::dvec3 posNorm = cs::utils::convert::toCartesian(l, radii, h);

    averagePositionNorm += posNorm / static_cast<double>(mPoints.size());
  }

  mGuiAnchor->setAnchorPosition(averagePositionNorm);
  mPlaneAnchor->setAnchorPosition(averagePositionNorm);

  // This seems to be the first time the tool is moved, so we have to store the distance to the
  // observer so that we can scale the tool later based on the observer's position.
  if (pScaleDistance.get() < 0) {
    try {
      pScaleDistance = mSolarSystem->getObserver().getAnchorScale() *
                       glm::length(mSolarSystem->getObserver().getRelativePosition(
                           mTimeControl->pSimulationTime.get(), *mGuiAnchor));
    } catch (std::exception const& e) {
      // Getting the relative transformation may fail due to insufficient SPICE data.
      logger().warn("Failed to calculate scale distance of Dip and Strike Tool: {}", e.what());
    }
  }

  // calculate center of plane and normal
  // based on http://stackoverflow.com/questions/1400213/3d-least-squares-plane
  glm::dmat3 mat(0);
  glm::dvec3 vec(0);

  auto      center      = mPlaneAnchor->getAnchorPosition();
  glm::vec3 idealNormal = cs::utils::convert::cartesianToNormal(center, radii);
  mNormal               = idealNormal;
  mSize                 = 0;
  mOffset               = 0.F;

  for (auto const& p : mPoints) {
    glm::dvec2 l =
        cs::utils::convert::cartesianToLngLat(p->getAnchor()->getAnchorPosition(), radii);
    double     h       = body ? body->getHeight(l) : 0.0;
    glm::dvec3 posNorm = cs::utils::convert::toCartesian(l, radii, h);

    glm::dvec3 realtivePosition = posNorm - averagePositionNorm;

    mSize = std::max(mSize, glm::length(realtivePosition));

    mat[0][0] += realtivePosition.x * realtivePosition.x;
    mat[1][0] += realtivePosition.x * realtivePosition.y;
    mat[2][0] += realtivePosition.x;
    mat[0][1] += realtivePosition.x * realtivePosition.y;
    mat[1][1] += realtivePosition.y * realtivePosition.y;
    mat[2][1] += realtivePosition.y;
    mat[0][2] += realtivePosition.x;
    mat[1][2] += realtivePosition.y;
    mat[2][2] += 1;

    vec[0] += realtivePosition.x * realtivePosition.z;
    vec[1] += realtivePosition.y * realtivePosition.z;
    vec[2] += realtivePosition.z;
  }

  if (mPoints.size() > 2) {
    glm::vec3 solution = glm::inverse(mat) * vec;
    mNormal            = glm::normalize(glm::vec3(-solution.x, -solution.y, 1.F));
    mOffset            = solution.z;

    if (glm::dot(idealNormal, mNormal) < 0) {
      mNormal = -mNormal;
    }

    // calculate dip and strike directions
    glm::vec3 strike       = glm::normalize(glm::cross(mNormal, idealNormal));
    glm::vec3 dipDirection = glm::normalize(glm::cross(idealNormal, strike));
    mMip                   = glm::normalize(glm::cross(mNormal, strike));

    // calculate dip and strike values
    glm::vec3 north(0, 1, 0);
    float     fDip    = std::acos(glm::dot(mMip, dipDirection)) * 180 / glm::pi<float>();
    float     fStrike = std::acos(glm::dot(north, strike)) * 180 / glm::pi<float>();

    if (strike.x < 0) {
      fStrike = 360 - fStrike;
    }

    mGuiItem->callJavascript("setData", fDip, fStrike);
  } else {
    mMip = glm::normalize(glm::cross(mNormal, glm::vec3(0, 1, 0)));
    mGuiItem->callJavascript("setData", 0, 0);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void DipStrikeTool::update() {
  MultiPointTool::update();

  if (mVerticesDirty) {
    calculateDipAndStrike();
    mVerticesDirty = false;
  }

  double simulationTime(mTimeControl->pSimulationTime.get());

  cs::core::SolarSystem::scaleRelativeToObserver(*mGuiAnchor, mSolarSystem->getObserver(),
      simulationTime, pScaleDistance.get(), mSettings->mGraphics.pWorldUIScale.get());
  cs::core::SolarSystem::turnToObserver(
      *mGuiAnchor, mSolarSystem->getObserver(), simulationTime, false);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DipStrikeTool::Do() {
  glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT);

  // enable alpha blending for smooth line
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  std::array<GLfloat, 16> glMatMV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());

  glm::vec3 x(mMip);
  glm::vec3 y(mNormal);
  glm::vec3 z = glm::normalize(glm::cross(x, y));

  auto matMV = glm::make_mat4x4(glMatMV.data()) *
               glm::mat4(x.x, x.y, x.z, 0, y.x, y.y, y.z, 0, z.x, z.y, z.z, 0, 0, 0, mOffset, 1);

  matMV = glm::scale(matMV, glm::vec3(static_cast<float>(mSize) * pSize.get()));

  mShader.Bind();
  mVAO.Bind();
  glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glm::value_ptr(matMV));
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());
  mShader.SetUniform(mUniforms.opacity, pOpacity.get());
  mShader.SetUniform(mUniforms.farClip, cs::utils::getCurrentFarClipDistance());

  // draw the linestrip
  glDrawArrays(GL_TRIANGLE_FAN, 0, RESOLUTION + 1);
  mVAO.Release();
  mShader.Release();

  glPopAttrib();
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool DipStrikeTool::GetBoundingBox(VistaBoundingBox& bb) {
  std::array fMin{-0.1F, -0.1F, -0.1F};
  std::array fMax{0.1F, 0.1F, 0.1F};

  bb.SetBounds(fMin.data(), fMax.data());
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::measurementtools
