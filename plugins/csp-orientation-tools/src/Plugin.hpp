////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ORIENTATION_TOOLS_PLUGIN_HPP
#define CSP_ORIENTATION_TOOLS_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-graphics/ObjLoader.hpp"

#include <memory>
#include <optional>

namespace csp::orientationtools {

class Arrow;
class Axis;

/// Plugin that shows the orientation of objects by visualizing their coordinate systems
/// similar to modelling software like Blender.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {

    // Settings for a group of arrows.
    struct Arrows {
      float mSize;
      std::optional<bool> mDisableX;
      std::optional<bool> mDisableY;
      std::optional<bool> mDisableZ;
    };

    // Settings for a group of axes.
    struct Axis {
      float mSize;
      glm::vec3 mColor{};
      std::optional<bool> mDisableX;
      std::optional<bool> mDisableY;
      std::optional<bool> mDisableZ;
    };

    // All groups of arrows with their name as key.
    std::map<std::string, Arrows> mArrows;

    // All axes with their name as key.
    std::map<std::string, Axis> mAxes;

    cs::utils::DefaultProperty<bool> mEnableArrows{true};
    cs::utils::DefaultProperty<bool> mEnableAxes{true};
  };

  void init() override;
  void deInit() override;
  void update() override;

 private:
  void onLoad();

  // Creates and adds a group of arrows or axes for a setting.
  void addArrowsGroup(std::pair<const std::string, csp::orientationtools::Plugin::Settings::Arrows> settings);
  void addAxesGroup(std::pair<const std::string, csp::orientationtools::Plugin::Settings::Axis> settings);

  std::shared_ptr<Settings> mPluginSettings = std::make_shared<Settings>();

  std::map<std::string, std::vector<std::shared_ptr<Arrow>>> mArrows;
  std::map<std::string, std::vector<std::shared_ptr<Axis>>> mAxes;

  // Load the models used.
  std::shared_ptr<cs::graphics::ObjLoader> mArrowModel = std::make_shared<cs::graphics::ObjLoader>("../share/resources/models/arrow.obj");
  std::shared_ptr<cs::graphics::ObjLoader> mAxisModel = std::make_shared<cs::graphics::ObjLoader>("../share/resources/models/axis.obj");

  // Define colors for arrows.
  glm::vec4 mColorX{1.0f, 0.0f, 0.0f, 1.0f};
  glm::vec4 mColorY{0.0f, 1.0f, 0.0f, 1.0f};
  glm::vec4 mColorZ{0.0f, 0.0f, 1.0f, 1.0f};

  // Define the angles and rotation axes so the arrow base model (facing x) can look
  // in the corresponding direction.
  float mAngleX = 0.0f;
  float mAngleY = 90.0f;
  float mAngleZ = -90.0f;

  glm::dvec3 mRotAxisX{1.0f, 0.0f, 0.0f};
  glm::dvec3 mRotAxisY{0.0f, 0.0f, 1.0f};
  glm::dvec3 mRotAxisZ{0.0f, 1.0f, 0.0f};

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};

} // namespace csp::orientationtools

#endif // CSP_ORIENTATION_TOOLS_PLUGIN_HPP
