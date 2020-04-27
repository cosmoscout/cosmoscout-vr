////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_CORE_TOOLS_MULTIPOINT_TOOL_HPP
#define CS_CORE_TOOLS_MULTIPOINT_TOOL_HPP

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

namespace tools {

/// A base class for tools that need multiple points to work. It provides an interface for adding
/// points at the current pointer location and checking if a point is selected.
class CS_CORE_EXPORT MultiPointTool : public Tool {
 public:
  /// Public properties where external can connect slots to.
  cs::utils::Property<bool> pAddPointMode = false;

  /// Consider this to be read-only.
  cs::utils::Property<bool> pAnyPointSelected = false;

  /// All handels are drawn with this color.
  cs::utils::Property<glm::vec3> pColor = glm::vec3(0.75, 0.75, 1.0);

  /// Derived classes should set this to the initial distance of the tool to the observer when the
  /// tool is first updated. It will be used to scale the handles based on the current observer
  /// distance.
  cs::utils::Property<double> pScaleDistance = -1.0;

  MultiPointTool(std::shared_ptr<InputManager> pInputManager,
      std::shared_ptr<SolarSystem> pSolarSystem, std::shared_ptr<Settings> settings,
      std::shared_ptr<TimeControl> pTimeControl, std::string sCenter, std::string sFrame);

  MultiPointTool(MultiPointTool const& other) = delete;
  MultiPointTool(MultiPointTool&& other)      = delete;

  MultiPointTool& operator=(MultiPointTool const& other) = delete;
  MultiPointTool& operator=(MultiPointTool&& other) = delete;

  ~MultiPointTool() override;

  /// Called from Tools class.
  void update() override;

  /// Gets or sets the SPICE center name for all points.
  virtual void               setCenterName(std::string const& name);
  virtual std::string const& getCenterName() const;

  /// Gets or sets the SPICE frame name for all points.
  virtual void               setFrameName(std::string const& name);
  virtual std::string const& getFrameName() const;

  /// Use this to access all point positions at once.
  std::vector<glm::dvec2> getPositions() const;

  /// Use this to modify all point positions at once. onPointMoved() will be called if this leads to
  /// a position shift of a point; onPointAdded() and onPointRemoved() will be called respectively
  /// if the number of points changes.
  void setPositions(std::vector<glm::dvec2> const& positions);

  /// A derived class may call this in order to add a new point at the given position. If no
  /// position is given, the current pointer position will be used.
  void addPoint(std::optional<glm::dvec2> const& lngLat = std::nullopt);

 protected:
  /// Derived classes should implement these - they will be called after the corresponding event
  /// happened.
  virtual void onPointMoved()            = 0;
  virtual void onPointAdded()            = 0;
  virtual void onPointRemoved(int index) = 0;

  std::shared_ptr<InputManager> mInputManager;
  std::shared_ptr<SolarSystem>  mSolarSystem;
  std::shared_ptr<Settings>     mSettings;
  std::shared_ptr<TimeControl>  mTimeControl;

  std::list<std::shared_ptr<DeletableMark>> mPoints;

 private:
  int         mLeftButtonConnection  = -1;
  int         mRightButtonConnection = -1;
  std::string mCenter, mFrame;
};

} // namespace tools
} // namespace cs::core

#endif // CS_CORE_TOOLS_MULTIPOINT_TOOL_HPP
