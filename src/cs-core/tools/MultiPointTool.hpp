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
  cs::utils::Property<bool> pAddPointMode = true;

  /// Consider this to be read-only.
  cs::utils::Property<bool> pAnyPointSelected = false;

  MultiPointTool(std::shared_ptr<InputManager> const& pInputManager,
      std::shared_ptr<SolarSystem> const& pSolarSystem, std::shared_ptr<Settings> const& settings,
      std::shared_ptr<TimeControl> const& pTimeControl, std::string const& sCenter,
      std::string const& sFrame);

  virtual ~MultiPointTool();

  /// Called from Tools class.
  void update() override;

  /// Returns the SPICE center name of the celestial body this is attached to.
  std::string const& getCenterName() const;

  /// Returns the SPICE frame name of the celestial body this is attached to.
  std::string const& getFrameName() const;

 protected:
  /// Derived classes should implement these - they will be called after the corresponding event
  /// happened.
  virtual void onPointMoved()            = 0;
  virtual void onPointAdded()            = 0;
  virtual void onPointRemoved(int index) = 0;

  /// A derived class may call this in order to add a new point at the current pointer position.
  void addPoint();

  std::shared_ptr<InputManager> mInputManager;
  std::shared_ptr<SolarSystem>  mSolarSystem;
  std::shared_ptr<Settings>     mSettings;
  std::shared_ptr<TimeControl>  mTimeControl;

  std::list<std::shared_ptr<DeletableMark>> mPoints;

 private:
  int         mLeftButtonConnection = -1, mRightButtonConnection = -1;
  std::string mCenter, mFrame;
};

} // namespace tools
} // namespace cs::core

#endif // CS_CORE_TOOLS_MULTIPOINT_TOOL_HPP
