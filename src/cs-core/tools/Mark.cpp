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

namespace cs::core::tools {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string Mark::SHADER_VERT = R"(
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

const std::string Mark::SHADER_FRAG = R"(
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

Mark::Mark(std::shared_ptr<InputManager> const& pInputManager,
    std::shared_ptr<SolarSystem> const& pSolarSystem, std::shared_ptr<Settings> const& settings,
    std::shared_ptr<TimeControl> const& pTimeControl, std::string const& sCenter,
    std::string const& sFrame)
    : mInputManager(pInputManager)
    , mSolarSystem(pSolarSystem)
    , mSettings(settings)
    , mTimeControl(pTimeControl)
    , mVAO(new VistaVertexArrayObject())
    , mVBO(new VistaBufferObject())
    , mIBO(new VistaBufferObject())
    , mShader(new VistaGLSLShader()) {

  initData(sCenter, sFrame);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Mark::Mark(Mark const& other)
    : pLngLat(other.pLngLat)
    , pHovered(other.pHovered)
    , pSelected(other.pSelected)
    , pActive(other.pActive)
    , mInputManager(other.mInputManager)
    , mSolarSystem(other.mSolarSystem)
    , mSettings(other.mSettings)
    , mTimeControl(other.mTimeControl)
    , mOriginalDistance(other.mOriginalDistance)
    , mVAO(new VistaVertexArrayObject())
    , mVBO(new VistaBufferObject())
    , mIBO(new VistaBufferObject())
    , mShader(new VistaGLSLShader()) {

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
      mOriginalDistance, mSettings->mGraphics.pWidgetScale.get());
  SolarSystem::turnToObserver(*mAnchor, mSolarSystem->getObserver(), simulationTime, false);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Mark::Do() {
  GLfloat glMatMV[16], glMatP[16];
  glGetFloatv(GL_MODELVIEW_MATRIX, &glMatMV[0]);
  glGetFloatv(GL_PROJECTION_MATRIX, &glMatP[0]);

  mShader->Bind();
  mVAO->Bind();
  glUniformMatrix4fv(mShader->GetUniformLocation("uMatModelView"), 1, GL_FALSE, glMatMV);
  glUniformMatrix4fv(mShader->GetUniformLocation("uMatProjection"), 1, GL_FALSE, glMatP);
  mShader->SetUniform(mShader->GetUniformLocation("uHoverSelectActive"), pHovered.get() ? 1.f : 0.f,
      pSelected.get() ? 1.f : 0.f, pActive.get() ? 1.f : 0.f);
  mShader->SetUniform(
      mShader->GetUniformLocation("uFarClip"), cs::utils::getCurrentFarClipDistance());
  mShader->SetUniform(
      mShader->GetUniformLocation("uColor"), pColor.get().x, pColor.get().y, pColor.get().z);

  glDrawElements(GL_TRIANGLES, (GLsizei)mIndexCount, GL_UNSIGNED_INT, nullptr);
  mVAO->Release();
  mShader->Release();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Mark::GetBoundingBox(VistaBoundingBox& bb) {
  float fMin[3] = {-0.01f, -0.01f, -0.01f};
  float fMax[3] = {0.01f, 0.01f, 0.01f};

  bb.SetBounds(fMin, fMax);

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Mark::initData(std::string const& sCenter, std::string const& sFrame) {
  mShader->InitVertexShaderFromString(SHADER_VERT);
  mShader->InitFragmentShaderFromString(SHADER_FRAG);
  mShader->Link();

  auto pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  mAnchor = std::make_shared<cs::scene::CelestialAnchorNode>(
      pSG->GetRoot(), pSG->GetNodeBridge(), "", sCenter, sFrame);
  mSolarSystem->registerAnchor(mAnchor);

  mParent = pSG->NewOpenGLNode(mAnchor.get(), this);
  mInputManager->registerSelectable(mParent);

  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mAnchor.get(), static_cast<int>(cs::utils::DrawOrder::eOpaqueItems));

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
            auto lngLatHeight = cs::utils::convert::toLngLatHeight(
                i.mPosition, body->getRadii()[0], body->getRadii()[0]);
            pLngLat = lngLatHeight.xy();
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
        lngLat, radii[0], radii[0], height * mSettings->mGraphics.pHeightScale.get());
    mAnchor->setAnchorPosition(cart);

    // This seems to be the first time the tool is moved, so we have to store the distance to the
    // observer so that we can scale the tool later based on the observer's position.
    if (mOriginalDistance < 0) {
      double simulationTime(mTimeControl->pSimulationTime.get());
      mOriginalDistance =
          mSolarSystem->getObserver().getAnchorScale() *
          glm::length(mSolarSystem->getObserver().getRelativePosition(simulationTime, *mAnchor));
    }
  });

  // connect the heightscale value to this object. Whenever the heightscale value changes
  // the landmark will be set to the correct height value
  mHeightScaleConnection = mSettings->mGraphics.pHeightScale.connect([this](float h) {
    auto   body   = mSolarSystem->getBody(mAnchor->getCenterName());
    double height = body->getHeight(pLngLat.get()) * h;
    auto   radii  = body->getRadii();
    auto   cart   = cs::utils::convert::toCartesian(pLngLat.get(), radii[0], radii[0], height);
    mAnchor->setAnchorPosition(cart);
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core::tools
