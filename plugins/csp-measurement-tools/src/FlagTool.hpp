////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_MEASUREMENT_TOOLS_FLAG_HPP
#define CSP_MEASUREMENT_TOOLS_FLAG_HPP

#include "../../../src/cs-core/tools/Mark.hpp"

namespace cs::gui {
class GuiItem;
class WorldSpaceGuiArea;
} // namespace cs::gui

class VistaTransformNode;

namespace csp::measurementtools {

/// A flag that can be planted and moved on the surface. It displays the current geographic
/// coordinates and height as well as the current address as precise as possible. You can also
/// edit the name of the flag.
class FlagTool : public cs::core::tools::Mark {
 public:
  /// This text is shown on the ui and can be edited by the user.
  cs::utils::Property<std::string> pText      = std::string("Flag");
  cs::utils::Property<bool>        pMinimized = false;

  FlagTool(std::shared_ptr<cs::core::InputManager> const& pInputManager,
      std::shared_ptr<cs::core::SolarSystem> const&       pSolarSystem,
      std::shared_ptr<cs::core::Settings> const&          settings,
      std::shared_ptr<cs::core::TimeControl> const& pTimeControl, std::string const& sCenter,
      std::string const& sFrame);

  FlagTool(FlagTool const& other) = delete;
  FlagTool(FlagTool&& other)      = delete;

  FlagTool& operator=(FlagTool const& other) = delete;
  FlagTool& operator=(FlagTool&& other) = delete;

  ~FlagTool() override;

  /// This is overwritten here as the flag should stand orthogonal on the planet's surface, rather
  /// than facing always the observer.
  void update() override;

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
