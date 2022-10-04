////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ANCHOR_LABELS_ANCHOR_LABEL_HPP
#define CSP_ANCHOR_LABELS_ANCHOR_LABEL_HPP

#include "../../../src/cs-scene/CelestialObject.hpp"
#include "../../../src/cs-utils/Property.hpp"
#include "Plugin.hpp"

class VistaOpenGLNode;
class VistaTransformNode;

namespace cs::scene {
class CelestialObject;
} // namespace cs::scene

namespace cs::gui {
class WorldSpaceGuiArea;
class GuiItem;
} // namespace cs::gui

namespace cs::core {
class SolarSystem;
class GuiManager;
class TimeControl;
class InputManager;
} // namespace cs::core

namespace csp::anchorlabels {
class AnchorLabel {
 public:
  AnchorLabel(std::string const& name, std::shared_ptr<const cs::scene::CelestialObject> object,
      std::shared_ptr<Plugin::Settings>       pluginSettings,
      std::shared_ptr<cs::core::SolarSystem>  solarSystem,
      std::shared_ptr<cs::core::GuiManager>   guiManager,
      std::shared_ptr<cs::core::InputManager> inputManager);

  ~AnchorLabel();

  AnchorLabel(AnchorLabel const& other) = delete;
  AnchorLabel(AnchorLabel&& other)      = delete;

  AnchorLabel& operator=(AnchorLabel const& other) = delete;
  AnchorLabel& operator=(AnchorLabel&& other) = delete;

  void update();

  std::shared_ptr<const cs::scene::CelestialObject> const& getObject() const;

  bool   shouldBeHidden() const;
  double bodySize() const;
  double distanceToCamera() const;

  void setSortKey(int key) const;

  void enable() const;
  void disable() const;

  glm::dvec4 getScreenSpaceBB() const;

 private:
  std::shared_ptr<const cs::scene::CelestialObject> mObject;

  std::shared_ptr<Plugin::Settings>       mPluginSettings;
  std::shared_ptr<cs::core::SolarSystem>  mSolarSystem;
  std::shared_ptr<cs::core::GuiManager>   mGuiManager;
  std::shared_ptr<cs::core::InputManager> mInputManager;

  std::unique_ptr<cs::gui::WorldSpaceGuiArea> mGuiArea;
  std::unique_ptr<cs::gui::GuiItem>           mGuiItem;
  std::unique_ptr<VistaOpenGLNode>            mGuiNode;
  std::unique_ptr<VistaTransformNode>         mObjectTransform;
  std::unique_ptr<VistaTransformNode>         mGuiTransform;

  glm::dvec3 mRelativeAnchorPosition{};
  int        mOffsetConnection = -1;
};
} // namespace csp::anchorlabels

#endif // CSP_ANCHOR_LABELS_ANCHOR_LABEL_HPP
