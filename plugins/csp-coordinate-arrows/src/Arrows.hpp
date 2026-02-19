////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_COORDINATE_ARROWS_ARROWS_HPP
#define CSP_COORDINATE_ARROWS_ARROWS_HPP

#include "Plugin.hpp"

#include "../../../src/cs-scene/CelestialObject.hpp"

#include <VistaBase/VistaColor.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <memory>

namespace csp::coordinatearrows {

class Arrows : public IVistaOpenGLDraw {
 public:
  Arrows(/*std::shared_ptr<Plugin::Settings> pluginSettings,
    std::shared_ptr<cs::core::SolarSystem>   solarSystem*/);

  Arrows(Arrows const& other) = delete;
  Arrows(Arrows&& other)      = delete;

  Arrows& operator=(Arrows const& other) = delete;
  Arrows& operator=(Arrows&& other)      = delete;

  ~Arrows() override;

  // This is called by the Plugin.
  void update(double tTime);

  // The arrows visualize the orientation of this object.
  //void setTargetName(std::string objectName);
  //std::string const& getTargetname() const;

  // The arrows are drawn relative to this object.
  //void setParentName(std::string objectname);
  //std::string const& getParentname() const;

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

  void setEnabled(bool value);

 private:
  std::shared_ptr<Plugin::Settings> mPluginSettings;
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;

  std::string mTargetName;
  std::string mParentName;

  std::vector<glm::dvec4> mPointsXArrow;
  std::vector<glm::dvec4> mPointsYArrow;
  std::vector<glm::dvec4> mPointsZArrow;
  double mArrowLength;

  bool mEnabled;

};

} // namespace csp::coordinatearrows

#endif // CSP_TRAJECTORIES_TRAJECTORY_HPP
