////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_TOOLS_MARK_HPP
#define CSL_TOOLS_MARK_HPP

#include "Tool.hpp"

#include <VistaBase/VistaColor.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>

namespace cs::scene {
class CelestialObject;
} // namespace cs::scene

class VistaTransformNode;
class VistaVertexArrayObject;

namespace cs::core {
class TimeControl;
class SolarSystem;
class InputManager;
class Settings;
} // namespace cs::core

namespace csl::tools {

/// A mark is a single point on the surface. It is selectable and draggable.
class CSL_TOOLS_EXPORT Mark : public IVistaOpenGLDraw, public Tool {
 public:
  /// Observable properties to get updates on state changes.
  cs::utils::Property<glm::dvec2> pLngLat   = glm::dvec2(0.0);
  cs::utils::Property<bool>       pHovered  = false;
  cs::utils::Property<bool>       pSelected = false;
  cs::utils::Property<bool>       pActive   = false;
  cs::utils::Property<glm::vec3>  pColor    = glm::vec3(0.75, 0.75, 1.0);

  /// This should be set to the initial distance of the tool to the observer. It will be used to
  /// scale the tool based on the current observer distance.
  cs::utils::Property<double> pScaleDistance = -1.0;

  Mark(std::shared_ptr<cs::core::InputManager> pInputManager,
      std::shared_ptr<cs::core::SolarSystem>   pSolarSystem,
      std::shared_ptr<cs::core::Settings> Settings, std::string objectName);

  Mark(Mark const& other);
  Mark(Mark&& other) = default;

  Mark& operator=(Mark const& other) = delete;
  Mark& operator=(Mark&& other)      = delete;

  ~Mark() override;

  glm::dvec3 const& getPosition() const;

  /// Called from Tools class.
  void update() override;

  /// Inherited from IVistaOpenGLDraw.
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 protected:
  std::shared_ptr<cs::core::InputManager> mInputManager;
  std::shared_ptr<cs::core::SolarSystem>  mSolarSystem;
  std::shared_ptr<cs::core::Settings>     mSettings;

  std::unique_ptr<VistaTransformNode> mTransform;
  std::unique_ptr<VistaOpenGLNode>    mParent;

 private:
  void initData();

  glm::dvec3 mPosition;

  std::unique_ptr<VistaVertexArrayObject> mVAO;
  std::unique_ptr<VistaBufferObject>      mVBO;
  std::unique_ptr<VistaBufferObject>      mIBO;
  std::unique_ptr<VistaGLSLShader>        mShader;

  struct {
    uint32_t modelViewMatrix   = 0;
    uint32_t projectionMatrix  = 0;
    uint32_t hoverSelectActive = 0;
    uint32_t color             = 0;
  } mUniforms;

  size_t mIndexCount{};

  int mSelfLngLatConnection = -1, mHoveredNodeConnection = -1, mSelectedNodeConnection = -1,
      mButtonsConnection = -1, mHoveredPlanetConnection = -1, mHeightScaleConnection = -1;
};

} // namespace csl::tools

#endif // CSL_TOOLS_MARK_HPP
