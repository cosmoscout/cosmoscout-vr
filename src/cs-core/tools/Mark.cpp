////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Mark.hpp"

#include "../../cs-scene/CelestialAnchorNode.hpp"
#include "../../cs-utils/convert.hpp"
#include "../../cs-utils/utils.hpp"
#include "../GuiManager.hpp"
#include "../InputManager.hpp"
#include "../Settings.hpp"
#include "../SolarSystem.hpp"
#include "../TimeControl.hpp"

#include <VistaDataFlowNet/VdfnObjectRegistry.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include <glm/gtc/type_ptr.hpp>
#include <utility>

namespace cs::core::tools {

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* SHADER_VERT = R"(
#version 330

layout(location=0) in vec3 iPosition;

out vec3 vPosition;
out vec3 vNormal;

uniform mat4 uMatModelView;
uniform mat4 uMatProjection;

void main()
{
    vPosition   = (uMatModelView * vec4(iPosition*0.005, 1.0)).xyz;
    gl_Position = uMatProjection * vec4(vPosition, 1.0);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* SHADER_FRAG = R"(
#version 330

uniform vec3 uHoverSelectActive;
uniform float uFarClip;
uniform vec3 uColor;

in vec3 vPosition;

layout(location = 0) out vec3 oColor;

void main()
{
    oColor = uColor;
    if (uHoverSelectActive.x > 0) oColor = mix(oColor, vec3(1, 1, 1), 0.2);
    if (uHoverSelectActive.y > 0) oColor = mix(oColor, vec3(1, 1, 1), 0.5);
    if (uHoverSelectActive.z > 0) oColor = mix(oColor, vec3(1, 1, 1), 0.8);

    // linearize depth value
    gl_FragDepth = length(vPosition) / uFarClip;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

Mark::Mark(std::shared_ptr<InputManager> pInputManager, std::shared_ptr<SolarSystem> pSolarSystem,
    std::shared_ptr<Settings> settings, std::shared_ptr<TimeControl> pTimeControl,
    std::string const& sCenter, std::string const& sFrame)
    : mInputManager(std::move(pInputManager))
    , mSolarSystem(std::move(pSolarSystem))
    , mSettings(std::move(settings))
    , mTimeControl(std::move(pTimeControl))
    , mVAO(std::make_unique<VistaVertexArrayObject>())
    , mVBO(std::make_unique<VistaBufferObject>())
    , mIBO(std::make_unique<VistaBufferObject>())
    , mShader(std::make_unique<VistaGLSLShader>()) {

  initData(sCenter, sFrame);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Mark::Mark(Mark const& other)
    : IVistaOpenGLDraw(other)
    , Tool(other)
    , pLngLat(other.pLngLat)
    , pHovered(other.pHovered)
    , pSelected(other.pSelected)
    , pActive(other.pActive)
    , pScaleDistance(other.pScaleDistance)
    , mInputManager(other.mInputManager)
    , mSolarSystem(other.mSolarSystem)
    , mSettings(other.mSettings)
    , mTimeControl(other.mTimeControl)
    , mVAO(std::make_unique<VistaVertexArrayObject>())
    , mVBO(std::make_unique<VistaBufferObject>())
    , mIBO(std::make_unique<VistaBufferObject>())
    , mShader(std::make_unique<VistaGLSLShader>()) {

  initData(other.getAnchor()->getCenterName(), other.getAnchor()->getFrameName());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Mark::~Mark() {
  mInputManager->pHoveredNode.disconnect(mHoveredNodeConnection);
  mInputManager->pSelectedNode.disconnect(mSelectedNodeConnection);
  mInputManager->pButtons[0].disconnect(mButtonsConnection);
  mInputManager->pHoveredObject.disconnect(mHoveredPlanetConnection);
  mSettings->mGraphics.pHeightScale.disconnect(mHeightScaleConnection);

  mInputManager->pHoveredNode    = nullptr;
  mInputManager->pHoveredGuiItem = nullptr;

  if (mParent) {
    mInputManager->unregisterSelectable(mParent);
    delete mParent;
  }

  if (mAnchor) {
    mSolarSystem->unregisterAnchor(mAnchor);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<cs::scene::CelestialAnchorNode> const& Mark::getAnchor() const {
  return mAnchor;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<cs::scene::CelestialAnchorNode>& Mark::getAnchor() {
  return mAnchor;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Mark::update() {
  double simulationTime(mTimeControl->pSimulationTime.get());

  SolarSystem::scaleRelativeToObserver(*mAnchor, mSolarSystem->getObserver(), simulationTime,
      pScaleDistance.get(), mSettings->mGraphics.pWorldUIScale.get());
  SolarSystem::turnToObserver(*mAnchor, mSolarSystem->getObserver(), simulationTime, false);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Mark::Do() {
  std::array<GLfloat, 16> glMatMV{};
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMatMV.data());
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());

  mShader->Bind();
  mVAO->Bind();
  glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glMatMV.data());
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());
  mShader->SetUniform(mUniforms.hoverSelectActive, pHovered.get() ? 1.F : 0.F,
      pSelected.get() ? 1.F : 0.F, pActive.get() ? 1.F : 0.F);
  mShader->SetUniform(mUniforms.farClip, cs::utils::getCurrentFarClipDistance());
  mShader->SetUniform(mUniforms.color, pColor.get().x, pColor.get().y, pColor.get().z);

  glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(mIndexCount), GL_UNSIGNED_INT, nullptr);
  mVAO->Release();
  mShader->Release();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Mark::GetBoundingBox(VistaBoundingBox& bb) {
  float extend = 0.01F;

  std::array fMin{-extend, -extend, -extend};
  std::array fMax{extend, extend, extend};

  bb.SetBounds(fMin.data(), fMax.data());

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Mark::initData(std::string const& sCenter, std::string const& sFrame) {
  mShader->InitVertexShaderFromString(SHADER_VERT);
  mShader->InitFragmentShaderFromString(SHADER_FRAG);
  mShader->Link();

  mUniforms.modelViewMatrix   = mShader->GetUniformLocation("uMatModelView");
  mUniforms.projectionMatrix  = mShader->GetUniformLocation("uMatProjection");
  mUniforms.hoverSelectActive = mShader->GetUniformLocation("uHoverSelectActive");
  mUniforms.farClip           = mShader->GetUniformLocation("uFarClip");
  mUniforms.color             = mShader->GetUniformLocation("uColor");

  auto* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  mAnchor = std::make_shared<cs::scene::CelestialAnchorNode>(
      pSG->GetRoot(), pSG->GetNodeBridge(), "", sCenter, sFrame);
  mSolarSystem->registerAnchor(mAnchor);

  mParent = pSG->NewOpenGLNode(mAnchor.get(), this);
  mInputManager->registerSelectable(mParent);

  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mAnchor.get(), static_cast<int>(cs::utils::DrawOrder::eOpaqueNonHDR));

  const std::array<glm::vec3, 26> POSITIONS = {glm::vec3(1, -1, 1), glm::vec3(-1, -1, -1),
      glm::vec3(1, -1, -1), glm::vec3(-1, 1, -1), glm::vec3(1, 1, 1), glm::vec3(1, 1, -1),
      glm::vec3(1, 1, -1), glm::vec3(1, -1, 1), glm::vec3(1, -1, -1), glm::vec3(1, 1, 1),
      glm::vec3(-1, -1, 1), glm::vec3(1, -1, 1), glm::vec3(-1, -1, 1), glm::vec3(-1, 1, -1),
      glm::vec3(-1, -1, -1), glm::vec3(1, -1, -1), glm::vec3(-1, 1, -1), glm::vec3(1, 1, -1),
      glm::vec3(-1, -1, 1), glm::vec3(-1, 1, 1), glm::vec3(1, 1, -1), glm::vec3(1, 1, 1),
      glm::vec3(1, -1, 1), glm::vec3(-1, 1, 1), glm::vec3(-1, 1, 1), glm::vec3(-1, -1, -1)};

  const std::array<uint32_t, 36> INDICES = {
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7,
      8,
      9,
      10,
      11,
      12,
      13,
      14,
      15,
      16,
      17,
      0,
      18,
      1,
      3,
      19,
      4,
      20,
      21,
      22,
      9,
      23,
      10,
      12,
      24,
      13,
      15,
      25,
      16,
  };

  mIndexCount = INDICES.size();

  mIBO->Bind(GL_ELEMENT_ARRAY_BUFFER);
  mIBO->BufferData(INDICES.size() * sizeof(uint32_t), INDICES.data(), GL_STATIC_DRAW);
  mIBO->Release();
  mVAO->SpecifyIndexBufferObject(mIBO.get());

  mVBO->Bind(GL_ARRAY_BUFFER);
  mVBO->BufferData(POSITIONS.size() * sizeof(glm::vec3), POSITIONS.data(), GL_STATIC_DRAW);
  mVBO->Release();

  mVAO->EnableAttributeArray(0);
  mVAO->SpecifyAttributeArrayFloat(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0, mVBO.get());

  // update hover state
  mHoveredNodeConnection = mInputManager->pHoveredNode.connect([this](IVistaNode* node) {
    if (node == mParent && !pHovered.get()) {
      pHovered = true;
      cs::core::GuiManager::setCursor(cs::gui::Cursor::eHand);
    } else if (node != mParent && pHovered.get()) {
      pHovered = false;
      cs::core::GuiManager::setCursor(cs::gui::Cursor::ePointer);
    }
  });

  mSelectedNodeConnection = mInputManager->pSelectedNode.connect(
      [this](IVistaNode* node) { pSelected = (node == mParent); });

  mButtonsConnection = mInputManager->pButtons[0].connect(
      [this](bool press) { pActive = (press && pHovered.get()); });

  mHoveredPlanetConnection =
      mInputManager->pHoveredObject.connect([this](InputManager::Intersection const& i) {
        if (pActive.get() && i.mObject) {
          auto body = std::dynamic_pointer_cast<cs::scene::CelestialBody>(i.mObject);
          if (body && body->getCenterName() == mAnchor->getCenterName()) {
            auto radii = body->getRadii();
            pLngLat    = cs::utils::convert::cartesianToLngLat(i.mPosition, radii);
          }
        }
      });

  // update position
  mSelfLngLatConnection = pLngLat.connect([this](glm::dvec2 const& lngLat) {
    // Request the height under the Mark and add it
    auto   body   = mSolarSystem->getBody(mAnchor->getCenterName());
    double height = body->getHeight(lngLat);
    auto   radii  = body->getRadii();
    auto   cart   = cs::utils::convert::toCartesian(
        lngLat, radii, height * mSettings->mGraphics.pHeightScale.get());
    mAnchor->setAnchorPosition(cart);
  });

  // connect the heightscale value to this object. Whenever the heightscale value changes
  // the landmark will be set to the correct height value
  mHeightScaleConnection = mSettings->mGraphics.pHeightScale.connect([this](float h) {
    auto   body   = mSolarSystem->getBody(mAnchor->getCenterName());
    double height = body->getHeight(pLngLat.get()) * h;
    auto   radii  = body->getRadii();
    auto   cart   = cs::utils::convert::toCartesian(pLngLat.get(), radii, height);
    mAnchor->setAnchorPosition(cart);
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core::tools
