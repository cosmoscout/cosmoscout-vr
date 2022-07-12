////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Mark.hpp"

#include "../../cs-scene/CelestialObject.hpp"
#include "../../cs-scene/CelestialSurface.hpp"
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
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include <glm/gtc/type_ptr.hpp>
#include <utility>

namespace cs::core::tools {

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* SHADER_VERT = R"(
#version 330

layout(location=0) in vec3 iPosition;

uniform mat4 uMatModelView;
uniform mat4 uMatProjection;

void main()
{
    vec3 pos    = (uMatModelView * vec4(iPosition*0.005, 1.0)).xyz;
    gl_Position = uMatProjection * vec4(pos, 1.0);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* SHADER_FRAG = R"(
#version 330

uniform vec3 uHoverSelectActive;
uniform vec3 uColor;

layout(location = 0) out vec3 oColor;

void main()
{
    oColor = uColor;
    if (uHoverSelectActive.x > 0) oColor = mix(oColor, vec3(1, 1, 1), 0.2);
    if (uHoverSelectActive.y > 0) oColor = mix(oColor, vec3(1, 1, 1), 0.5);
    if (uHoverSelectActive.z > 0) oColor = mix(oColor, vec3(1, 1, 1), 0.8);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

Mark::Mark(std::shared_ptr<InputManager> pInputManager, std::shared_ptr<SolarSystem> pSolarSystem,
    std::shared_ptr<Settings> settings, std::string const& objectName)
    : mInputManager(std::move(pInputManager))
    , mSolarSystem(std::move(pSolarSystem))
    , mSettings(std::move(settings))
    , mObject(mSolarSystem->getObject(objectName))
    , mPosition(0.0, 0.0, 0.0)
    , mRotation(1.0, 0.0, 0.0, 0.0)
    , mVAO(std::make_unique<VistaVertexArrayObject>())
    , mVBO(std::make_unique<VistaBufferObject>())
    , mIBO(std::make_unique<VistaBufferObject>())
    , mShader(std::make_unique<VistaGLSLShader>()) {

  initData();
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
    , mObject(other.mObject)
    , mVAO(std::make_unique<VistaVertexArrayObject>())
    , mVBO(std::make_unique<VistaBufferObject>())
    , mIBO(std::make_unique<VistaBufferObject>())
    , mShader(std::make_unique<VistaGLSLShader>()) {

  initData();
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
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Mark::update() {
  mScale = mSolarSystem->getScaleBasedOnObserverDistance(
      mObject, mPosition, pScaleDistance.get(), mSettings->mGraphics.pWorldUIScale.get());

  mRotation = mSolarSystem->getRotationToObserver(mObject, mPosition, false);
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

void Mark::initData() {
  mShader->InitVertexShaderFromString(SHADER_VERT);
  mShader->InitFragmentShaderFromString(SHADER_FRAG);
  mShader->Link();

  mUniforms.modelViewMatrix   = mShader->GetUniformLocation("uMatModelView");
  mUniforms.projectionMatrix  = mShader->GetUniformLocation("uMatProjection");
  mUniforms.hoverSelectActive = mShader->GetUniformLocation("uHoverSelectActive");
  mUniforms.color             = mShader->GetUniformLocation("uColor");

  auto* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  mTransform = pSG->NewTransformNode(pSG->GetRoot());
  mParent    = pSG->NewOpenGLNode(mTransform, this);
  mInputManager->registerSelectable(mParent);

  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mTransform, static_cast<int>(cs::utils::DrawOrder::eOpaqueNonHDR));

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
          if (i.mObject == mObject) {
            auto radii = mObject->getRadii();
            pLngLat    = cs::utils::convert::cartesianToLngLat(i.mPosition, radii);
          }
        }
      });

  // update position
  mSelfLngLatConnection = pLngLat.connect([this](glm::dvec2 const& lngLat) {
    // Request the height under the Mark and add it
    auto   surface = mObject->getSurface();
    double height  = surface ? surface->getHeight(lngLat) : 0.0;
    auto   radii   = mObject->getRadii();
    mPosition      = cs::utils::convert::toCartesian(
        lngLat, radii, height * mSettings->mGraphics.pHeightScale.get());
  });

  // connect the heightscale value to this object. Whenever the heightscale value changes
  // the landmark will be set to the correct height value
  mHeightScaleConnection = mSettings->mGraphics.pHeightScale.connect([this](float h) {
    auto   surface = mObject->getSurface();
    double height  = surface ? surface->getHeight(pLngLat.get()) * h : 0.0;
    auto   radii   = mObject->getRadii();
    mPosition      = cs::utils::convert::toCartesian(pLngLat.get(), radii, height);
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core::tools
