////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_ANCHOR_LABELS_ANCHOR_LABEL_HPP
#define CSP_ANCHOR_LABELS_ANCHOR_LABEL_HPP

#include "../../../src/cs-scene/CelestialBody.hpp"
#include "../../../src/cs-utils/Property.hpp"
#include "Plugin.hpp"

class VistaOpenGLNode;
class VistaTransformNode;

namespace cs::scene {
class CelestialBody;
class CelestialAnchorNode;
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
  AnchorLabel(cs::scene::CelestialBody const* body,
      std::shared_ptr<Plugin::Settings>       pluginSettings,
      std::shared_ptr<cs::core::SolarSystem>  solarSystem,
      std::shared_ptr<cs::core::GuiManager>   guiManager,
      std::shared_ptr<cs::core::TimeControl>  timeControl,
      std::shared_ptr<cs::core::InputManager> inputManager);

  ~AnchorLabel();

  AnchorLabel(AnchorLabel const& other) = delete;
  AnchorLabel(AnchorLabel&& other)      = delete;

  AnchorLabel& operator=(AnchorLabel const& other) = delete;
  AnchorLabel& operator=(AnchorLabel&& other) = delete;

  void update();

  std::string const& getCenterName() const;

  bool   shouldBeHidden() const;
  double bodySize() const;
  double distanceToCamera() const;

  void setSortKey(int key) const;

  void enable() const;
  void disable() const;

  glm::dvec4 getScreenSpaceBB() const;

 private:
  cs::scene::CelestialBody const* const mBody;

  std::shared_ptr<Plugin::Settings>       mPluginSettings;
  std::shared_ptr<cs::core::SolarSystem>  mSolarSystem;
  std::shared_ptr<cs::core::GuiManager>   mGuiManager;
  std::shared_ptr<cs::core::TimeControl>  mTimeControl;
  std::shared_ptr<cs::core::InputManager> mInputManager;

  std::shared_ptr<cs::scene::CelestialAnchorNode> mAnchor;

  std::unique_ptr<cs::gui::WorldSpaceGuiArea> mGuiArea;
  std::unique_ptr<cs::gui::GuiItem>           mGuiItem;
  std::unique_ptr<VistaOpenGLNode>            mGuiNode;
  std::unique_ptr<VistaTransformNode>         mGuiTransform;

  glm::dvec3 mRelativeAnchorPosition{};
  int        mOffsetConnection = -1;
};
} // namespace csp::anchorlabels

#endif // CSP_ANCHOR_LABELS_ANCHOR_LABEL_HPP
