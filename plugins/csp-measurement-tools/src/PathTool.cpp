////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "PathTool.hpp"

#include "logger.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/InputManager.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"
#include "../../../src/cs-core/tools/DeletableMark.hpp"
#include "../../../src/cs-scene/CelestialSurface.hpp"
#include "../../../src/cs-utils/convert.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include <glm/gtc/type_ptr.hpp>

namespace csp::measurementtools {

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* PathTool::SHADER_VERT = R"(
#version 330

layout(location=0) in vec3 iPosition;

uniform mat4 uMatModelView;
uniform mat4 uMatProjection;

void main()
{
    vec4 pos    = uMatModelView * vec4(iPosition, 1.0);
    gl_Position = uMatProjection * pos;
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* PathTool::SHADER_FRAG = R"(
#version 330

uniform vec3 uColor;

layout(location = 0) out vec4 oColor;

void main()
{
    oColor = vec4(uColor, 1.0);
}
)";

////////////////////////////////////////////////////////////////////////////////////////////////////

PathTool::PathTool(std::shared_ptr<cs::core::InputManager> pInputManager,
    std::shared_ptr<cs::core::SolarSystem>                 pSolarSystem,
    std::shared_ptr<cs::core::Settings> settings, std::string objectName)
    : MultiPointTool(std::move(pInputManager), std::move(pSolarSystem), std::move(settings),
          std::move(objectName))
    , mGuiArea(std::make_unique<cs::gui::WorldSpaceGuiArea>(800, 475))
    , mGuiItem(
          std::make_unique<cs::gui::GuiItem>("file://{toolZoom}../share/resources/gui/path.html")) {

  // create the shader
  mShader.InitVertexShaderFromString(SHADER_VERT);
  mShader.InitFragmentShaderFromString(SHADER_FRAG);
  mShader.Link();

  mUniforms.modelViewMatrix  = mShader.GetUniformLocation("uMatModelView");
  mUniforms.projectionMatrix = mShader.GetUniformLocation("uMatProjection");
  mUniforms.color            = mShader.GetUniformLocation("uColor");

  // attach this as OpenGLNode to scenegraph's root (all line vertices
  // will be draw relative to the observer, therfore we do not want
  // any transformation)
  auto* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
  mPathOpenGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));

  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mPathOpenGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eOpaqueNonHDR));

  // create a VistaTransformNode for the user interface
  // it will be moved to the center of all points when a point is moved
  // and rotated in such a way, that it always faces the observer
  mGuiAnchor.reset(pSG->NewTransformNode(pSG->GetRoot()));

  // create the user interface
  mGuiTransform.reset(pSG->NewTransformNode(mGuiAnchor.get()));
  mGuiTransform->Translate(0.F, 0.9F, 0.F);
  mGuiTransform->Scale(0.0005F * static_cast<float>(mGuiArea->getWidth()),
      0.0005F * static_cast<float>(mGuiArea->getHeight()), 1.F);
  mGuiTransform->Rotate(VistaAxisAndAngle(VistaVector3D(0.0, 1.0, 0.0), -glm::pi<float>() / 2.F));
  mGuiArea->addItem(mGuiItem.get());
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

  mGuiItem->setCursorChangeCallback([](cs::gui::Cursor c) { cs::core::GuiManager::setCursor(c); });

  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mGuiAnchor.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems));

  // whenever the height scale changes our vertex positions need to be updated
  mScaleConnection = mSettings->mGraphics.pHeightScale.connectAndTouch(
      [this](float /*h*/) { mVerticesDirty = true; });

  // Update text.
  mTextConnection = pText.connectAndTouch(
      [this](std::string const& value) { mGuiItem->callJavascript("setText", value); });

  mGuiItem->registerCallback("onSetText",
      "This is called whenever the text input of the tool's name changes.",
      std::function(
          [this](std::string&& value) { pText.setWithEmitForAllButOne(value, mTextConnection); }));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

PathTool::~PathTool() {
  mSettings->mGraphics.pHeightScale.disconnect(mScaleConnection);
  mGuiItem->unregisterCallback("deleteMe");
  mGuiItem->unregisterCallback("setAddPointMode");
  mGuiItem->unregisterCallback("onSetText");

  mInputManager->unregisterSelectable(mGuiOpenGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PathTool::setNumSamples(int const& numSamples) {
  if (mNumSamples != numSamples) {
    mNumSamples    = numSamples;
    mVerticesDirty = true;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec4 PathTool::getInterpolatedPosBetweenTwoMarks(cs::core::tools::DeletableMark const& l0,
    cs::core::tools::DeletableMark const& l1, double value, double const& scale) {

  auto       object = mSolarSystem->getObject(getObjectName());
  glm::dvec3 radii  = object->getRadii();

  // Get cartesian coordinates for interpolation
  glm::dvec3 p0              = cs::utils::convert::toCartesian(l0.pLngLat.get(), radii, 0.0);
  glm::dvec3 p1              = cs::utils::convert::toCartesian(l1.pLngLat.get(), radii, 0.0);
  glm::dvec3 interpolatedPos = p0 + (value * (p1 - p0));

  // Calc final position
  glm::dvec2 ll     = cs::utils::convert::cartesianToLngLat(interpolatedPos, radii);
  double     height = (object->getSurface() ? object->getSurface()->getHeight(ll) : 0.0) * scale;
  glm::dvec3 pos    = cs::utils::convert::toCartesian(ll, radii, height);

  return glm::dvec4(pos, height);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PathTool::onPointMoved() {
  mVerticesDirty = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PathTool::onPointAdded() {
  mVerticesDirty = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PathTool::onPointRemoved(int /*index*/) {
  mVerticesDirty = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PathTool::updateLineVertices() {
  if (mPoints.empty()) {
    return;
  }

  // Fill the vertex buffer with sampled data
  mSampledPositions.clear();

  auto object = mSolarSystem->getObject(getObjectName());

  mPosition = glm::dvec3(0.0);
  for (auto const& mark : mPoints) {
    mPosition += mark->getPosition() / static_cast<double>(mPoints.size());
  }

  double heightScale = mSettings->mGraphics.pHeightScale.get();
  auto   lastMark    = mPoints.begin();
  auto   currMark    = ++mPoints.begin();

  std::stringstream json;
  std::string       jsonSeperator;
  double            distance = -1;
  glm::dvec3        lastPos(0.0);

  while (currMark != mPoints.end()) {
    // generate X points for each line segment
    for (int vertex_id = 0; vertex_id < mNumSamples; vertex_id++) {
      glm::dvec4 pos = getInterpolatedPosBetweenTwoMarks(
          **lastMark, **currMark, (vertex_id / static_cast<double>(mNumSamples)), heightScale);
      mSampledPositions.push_back(pos.xyz());

      // coordinate normalized by height scale; to count distance correctly
      glm::dvec4 posNorm = pos;
      if (heightScale != 1) {
        posNorm = getInterpolatedPosBetweenTwoMarks(
            **lastMark, **currMark, (vertex_id / static_cast<double>(mNumSamples)), 1);
      }

      if (distance < 0) {
        distance = 0;
      } else {
        distance += glm::length(posNorm.xyz() - lastPos);
      }

      json << jsonSeperator << "[" << distance << "," << pos.w / heightScale << "]";
      jsonSeperator = ",";

      lastPos = posNorm.xyz();
    }

    lastMark = currMark;
    ++currMark;
  }

  mGuiItem->callJavascript("setData", "[" + json.str() + "]");

  mIndexCount = mSampledPositions.size();

  // Upload new data
  mVBO.Bind(GL_ARRAY_BUFFER);
  mVBO.BufferData(mSampledPositions.size() * sizeof(glm::vec3), nullptr, GL_DYNAMIC_DRAW);
  mVBO.Release();

  mVAO.EnableAttributeArray(0);
  mVAO.SpecifyAttributeArrayFloat(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), 0, &mVBO);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void PathTool::update() {
  MultiPointTool::update();

  if (mVerticesDirty) {
    updateLineVertices();
    mVerticesDirty = false;
  }

  auto object = mSolarSystem->getObject(getObjectName());

  auto guiScale = mSolarSystem->getScaleBasedOnObserverDistance(
      object, mPosition, pScaleDistance.get(), mSettings->mGraphics.pWorldUIScale.get());
  auto guiRotation = mSolarSystem->getRotationToObserver(object, mPosition, false);

  auto guiTransform = object->getObserverRelativeTransform(mPosition, guiRotation, guiScale);

  mGuiAnchor->SetTransform(glm::value_ptr(guiTransform), true);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool PathTool::Do() {
  // transform all high precision sample points to observer centric
  // low precision coordinates
  std::vector<glm::vec3> vRelativePositions(mIndexCount);

  auto object    = mSolarSystem->getObject(getObjectName());
  auto transform = object->getObserverRelativeTransform(mPosition);

  for (size_t i(0); i < mIndexCount; ++i) {
    vRelativePositions[i] = (transform * glm::dvec4(mSampledPositions[i] - mPosition, 1.0)).xyz();
  }

  // upload the new points to the GPU
  mVBO.Bind(GL_ARRAY_BUFFER);
  mVBO.BufferSubData(0, vRelativePositions.size() * sizeof(glm::vec3), vRelativePositions.data());
  mVBO.Release();

  glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_LINE_BIT);

  // enable alpha blending for smooth line
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

  // draw the linestrip
  glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(mIndexCount));
  mVAO.Release();
  mShader.Release();

  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool PathTool::GetBoundingBox(VistaBoundingBox& bb) {
  std::array fMin{-0.1F, -0.1F, -0.1F};
  std::array fMax{0.1F, 0.1F, 0.1F};

  bb.SetBounds(fMin.data(), fMax.data());
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::measurementtools
