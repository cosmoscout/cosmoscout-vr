////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_TOOLS_DELETABLE_MARK_HPP
#define CSL_TOOLS_DELETABLE_MARK_HPP

#include "Mark.hpp"

namespace cs::gui {
class GuiItem;
class WorldSpaceGuiArea;
} // namespace cs::gui

namespace csl::tools {

/// A Mark with a delete symbol above when it is selected.
class CSL_TOOLS_EXPORT DeletableMark : public Mark {
 public:
  DeletableMark(std::shared_ptr<cs::core::InputManager> pInputManager,
      std::shared_ptr<cs::core::SolarSystem>            pSolarSystem,
      std::shared_ptr<cs::core::Settings> settings, std::string objectName);

  DeletableMark(DeletableMark const& other) = delete;
  DeletableMark(DeletableMark&& other)      = delete;

  DeletableMark& operator=(DeletableMark const& other) = delete;
  DeletableMark& operator=(DeletableMark&& other)      = delete;

  ~DeletableMark() override;

 private:
  void initData();

  VistaOpenGLNode*                            mGuiNode = nullptr;
  std::unique_ptr<cs::gui::WorldSpaceGuiArea> mGuiArea;
  std::unique_ptr<cs::gui::GuiItem>           mGuiItem;

  int mSelfSelectedConnection = -1;
};

} // namespace csl::tools

#endif // CSL_TOOLS_DELETABLE_MARK_HPP
