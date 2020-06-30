////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_CORE_DELETABLE_MARK_HPP
#define CS_CORE_DELETABLE_MARK_HPP

#include "Mark.hpp"

namespace cs::gui {
class GuiItem;
class WorldSpaceGuiArea;
} // namespace cs::gui

namespace cs::core::tools {

/// A Mark with a delete symbol above when it is selected.
class CS_CORE_EXPORT DeletableMark : public Mark {
 public:
  DeletableMark(std::shared_ptr<InputManager> const& pInputManager,
      std::shared_ptr<SolarSystem> const& pSolarSystem, std::shared_ptr<Settings> const& settings,
      std::shared_ptr<TimeControl> const& pTimeControl, std::string const& sCenter,
      std::string const& sFrame);

  DeletableMark(DeletableMark const& other) = delete;
  DeletableMark(DeletableMark&& other)      = delete;

  DeletableMark& operator=(DeletableMark const& other) = delete;
  DeletableMark& operator=(DeletableMark&& other) = delete;

  ~DeletableMark() override;

 private:
  void initData();

  VistaOpenGLNode*                        mGuiNode = nullptr;
  std::unique_ptr<gui::WorldSpaceGuiArea> mGuiArea;
  std::unique_ptr<gui::GuiItem>           mGuiItem;

  int mSelfSelectedConnection = -1;
};

} // namespace cs::core::tools

#endif // CS_CORE_DELETABLE_MARK_HPP
