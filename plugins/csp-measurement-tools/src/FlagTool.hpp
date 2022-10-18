////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_MEASUREMENT_TOOLS_FLAG_HPP
#define CSP_MEASUREMENT_TOOLS_FLAG_HPP

#include "../../csl-tools/src/Mark.hpp"

namespace cs::gui {
class GuiItem;
class WorldSpaceGuiArea;
} // namespace cs::gui

class VistaTransformNode;

namespace csp::measurementtools {

/// A flag that can be planted and moved on the surface. It displays the current geographic
/// coordinates and height as well as the current address as precise as possible. You can also
/// edit the name of the flag.
class FlagTool : public csl::tools::Mark {
 public:
  /// This text is shown on the ui and can be edited by the user.
  cs::utils::Property<std::string> pText      = std::string("Flag");
  cs::utils::Property<bool>        pMinimized = false;

  FlagTool(std::shared_ptr<cs::core::InputManager> pInputManager,
      std::shared_ptr<cs::core::SolarSystem>       pSolarSystem,
      std::shared_ptr<cs::core::Settings> settings, std::string objectName);

  FlagTool(FlagTool const& other) = delete;
  FlagTool(FlagTool&& other)      = delete;

  FlagTool& operator=(FlagTool const& other) = delete;
  FlagTool& operator=(FlagTool&& other) = delete;

  ~FlagTool() override;

 private:
  std::unique_ptr<VistaTransformNode>         mGuiTransform;
  std::unique_ptr<VistaOpenGLNode>            mGuiNode;
  std::unique_ptr<cs::gui::WorldSpaceGuiArea> mGuiArea;
  std::unique_ptr<cs::gui::GuiItem>           mGuiItem;

  int mTextConnection        = -1;
  int mDoubleClickConnection = -1;
};
} // namespace csp::measurementtools
#endif // CSP_MEASUREMENT_TOOLS_FLAG_HPP
