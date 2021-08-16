////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_USER_STUDY_STAGE_HPP
#define CSP_USER_STUDY_STAGE_HPP

#include "Plugin.hpp"

namespace cs::core {
class Settings;
} // namespace cs::core

class VistaOpenGLNode;
class VistaTransformNode;

namespace cs::scene {
class CelestialAnchorNode;
} // namespace cs::scene

namespace cs::gui {
class WorldSpaceGuiArea;
class GuiItem;
} // namespace cs::gui

namespace csp::userstudy {

/// The user study contains and displays a pre programmed sequence of stages.
/// Each stage consist of a Web View positioned at a bookmark, and it presents the user with a task
/// depending on the stage type. The scenario is configurable via the application config file. See
/// README.md for detailed information.
class Stage {
 public:
  Stage(csp::userstudy::Plugin::Settings::StageType type, std::shared_ptr<cs::scene::CelestialAnchorNode> anchor,
      float scale);

  Stage(Stage const& other) = delete;
  Stage(Stage&& other)      = default;

  Stage& operator=(Stage const& other) = delete;
  Stage& operator=(Stage&& other) = delete;

  ~Stage() = default;

 private:
  std::shared_ptr<cs::scene::CelestialAnchorNode> mAnchor;
  Plugin::Settings::StageType                     mType;
  float                                           mScale = 1.0F;
  std::unique_ptr<cs::gui::WorldSpaceGuiArea>     mGuiArea;
  std::unique_ptr<VistaTransformNode>             mTransform;
  std::unique_ptr<VistaOpenGLNode>                mGuiNode;
  std::unique_ptr<cs::gui::GuiItem>               mGuiItem;

}; // class Stage
} // namespace csp::userstudy

#endif // CSP_USER_STUDY_STAGE_HPP