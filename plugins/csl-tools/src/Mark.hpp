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
#include <VistaOGLExt/VistaGLSLShader.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>

namespace cs::scene {
class CelestialObject;
} // namespace cs::scene

class VistaTransformNode;

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
  enum class ElevationMode { eOverSurface, eOverZero };

  /// Observable properties to get updates on state changes. Consider these to be read-only.
  cs::utils::Property<bool> pHovered  = false;
  cs::utils::Property<bool> pSelected = false;
  cs::utils::Property<bool> pActive   = false;

  /// The position of the mark in longitude and latitude (in radians).
  cs::utils::Property<glm::dvec2> pLngLat = glm::dvec2(0.0);

  /// The elevation of the mark in meters. If this is set to 0, the mark will be placed on the
  /// surface. For positive values, the mark will be floating above the surface with a thin line
  /// connecting it to the surface.
  cs::utils::Property<double>        pElevation     = 0.0;
  cs::utils::Property<ElevationMode> pElevationMode = ElevationMode::eOverSurface;

  /// If this is true, the mark can be dragged around. You can connect to pLngLat to get updates on
  /// the position.
  cs::utils::Property<bool>      pDraggable = true;
  cs::utils::Property<glm::vec3> pColor     = glm::vec3(0.75, 0.75, 1.0);

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
  void updatePosition(
      glm::dvec2 const& lngLat, double elevation, ElevationMode mode, float heightScale);

  glm::dvec3                       mPosition;
  double                           mScale = 1.0;
  std::unique_ptr<VistaGLSLShader> mShader;

  struct {
    uint32_t modelViewMatrix   = 0;
    uint32_t projectionMatrix  = 0;
    uint32_t hoverSelectActive = 0;
    uint32_t scale             = 0;
    uint32_t offset            = 0;
    uint32_t color             = 0;
  } mUniforms;

  size_t mIndexCount{};

  int mHoveredNodeConnection = -1, mSelectedNodeConnection = -1, mButtonsConnection = -1,
      mHoveredPlanetConnection = -1, mHeightScaleConnection = -1;
};

} // namespace csl::tools

#endif // CSL_TOOLS_MARK_HPP
