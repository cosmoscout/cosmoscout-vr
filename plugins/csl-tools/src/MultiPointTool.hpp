////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_TOOLS_MULTIPOINT_TOOL_HPP
#define CSL_TOOLS_MULTIPOINT_TOOL_HPP

#include "DeletableMark.hpp"
#include "Tool.hpp"

#include <list>
#include <memory>
#include <optional>

namespace cs::core {
class TimeControl;
class SolarSystem;
class InputManager;
class Settings;
} // namespace cs::core

namespace csl::tools {

/// A base class for tools that need multiple points to work. It provides an interface for adding
/// points at the current pointer location and checking if a point is selected.
class CSL_TOOLS_EXPORT MultiPointTool : public Tool {
 public:
  /// Public properties where external can connect slots to.
  cs::utils::Property<bool> pAddPointMode = false;

  /// This is true if any of the tool's points has currently the selected state. Consider this to be
  /// read-only.
  cs::utils::Property<bool> pAnyPointSelected = false;

  /// This is true if any of the tool's points has currently the active state. This is usually the
  /// case if it is dragged around. Consider this to be read-only.
  cs::utils::Property<bool> pAnyPointActive = false;

  /// All handels are drawn with this color.
  cs::utils::Property<glm::vec3> pColor = glm::vec3(0.75, 0.75, 1.0);

  /// If this is true, the points of the tool can be dragged around.
  cs::utils::Property<bool> pPointsDraggable = true;

  /// If this is true, the points of the tool can be removed.
  cs::utils::Property<bool> pPointsDeletable = true;

  /// All points are drawn with this constant offset from the surface in meters.
  cs::utils::Property<double> pElevation = 0.0;

  /// Derived classes should set this to the initial distance of the tool to the observer when the
  /// tool is first updated. It will be used to scale the handles based on the current observer
  /// distance.
  cs::utils::Property<double> pScaleDistance = -1.0;

  MultiPointTool(std::shared_ptr<cs::core::InputManager> pInputManager,
      std::shared_ptr<cs::core::SolarSystem>             pSolarSystem,
      std::shared_ptr<cs::core::Settings> settings, std::string objectName);

  MultiPointTool(MultiPointTool const& other) = delete;
  MultiPointTool(MultiPointTool&& other)      = delete;

  MultiPointTool& operator=(MultiPointTool const& other) = delete;
  MultiPointTool& operator=(MultiPointTool&& other)      = delete;

  ~MultiPointTool() override;

  /// Called from Tools class.
  void update() override;

  // Assigns all points to a new celestial object.
  void setObjectName(std::string name) override;

  /// Use this to access all point positions at once.
  std::vector<glm::dvec2> getPositions() const;

  /// Use this to modify all point positions at once. onPointMoved() will be called if this leads to
  /// a position shift of a point; onPointAdded() and onPointRemoved() will be called respectively
  /// if the number of points changes.
  void setPositions(std::vector<glm::dvec2> const& positions);

  /// A derived class may call this in order to add a new point at the given position. If no
  /// position is given, the current pointer position will be used. If an index is given, the new
  /// point will be inserted at this position. Otherwise it will be appended to the end.
  void addPoint(std::optional<glm::dvec2> const& lngLat = std::nullopt,
      std::optional<size_t> const&               index  = std::nullopt);

 protected:
  /// Derived classes should implement these - they will be called after the corresponding event
  /// happened.
  virtual void onPointMoved()               = 0;
  virtual void onPointAdded(size_t index)   = 0;
  virtual void onPointRemoved(size_t index) = 0;

  std::shared_ptr<cs::core::InputManager> mInputManager;
  std::shared_ptr<cs::core::SolarSystem>  mSolarSystem;
  std::shared_ptr<cs::core::Settings>     mSettings;

  std::list<std::shared_ptr<DeletableMark>> mPoints;

 private:
  int mLeftButtonConnection  = -1;
  int mRightButtonConnection = -1;
};

} // namespace csl::tools

#endif // CSL_TOOLS_MULTIPOINT_TOOL_HPP
