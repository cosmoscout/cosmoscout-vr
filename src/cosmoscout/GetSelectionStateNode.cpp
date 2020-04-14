////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "GetSelectionStateNode.hpp"

#include "../cs-core/InputManager.hpp"
#include "../cs-gui/GuiItem.hpp"

#include <VistaAspects/VistaPropertyAwareable.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

GetSelectionStateNode::GetSelectionStateNode(
    cs::core::InputManager* pInputManager, VistaPropertyList const& /*unused*/)
    : mInputManager(pInputManager)
    , mSelectedGuiItemHasKeyboardFocus(new TVdfnPort<bool>())
    , mHoveredGuiItemAllowsScrolling(new TVdfnPort<bool>())
    , mHoveredGuiItem(new TVdfnPort<bool>())
    , mActiveGuiItem(new TVdfnPort<bool>())
    , mSelectedGuiItem(new TVdfnPort<bool>())
    , mHoveredNode(new TVdfnPort<bool>())
    , mActiveNode(new TVdfnPort<bool>())
    , mSelectedNode(new TVdfnPort<bool>()) {

  SetEvaluationFlag(true);

  // Ports will be deleted in the IVdfnNode's destructor.
  RegisterOutPort("selected_gui_item_has_keyboard_focus", mSelectedGuiItemHasKeyboardFocus);
  RegisterOutPort("hovered_gui_item_allows_scrolling", mHoveredGuiItemAllowsScrolling);
  RegisterOutPort("hovered_gui_item", mHoveredGuiItem);
  RegisterOutPort("active_gui_item", mActiveGuiItem);
  RegisterOutPort("selected_gui_item", mSelectedGuiItem);
  RegisterOutPort("hovered_node", mHoveredNode);
  RegisterOutPort("active_node", mActiveNode);
  RegisterOutPort("selected_node", mSelectedNode);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool GetSelectionStateNode::DoEvalNode() {

  const auto updatePort = [](TVdfnPort<bool>* port, bool newVal) {
    bool& oldVal = port->GetValueRef();
    if (oldVal != newVal) {
      oldVal = newVal;
      port->IncUpdateCounter();
    }
  };

  updatePort(mSelectedGuiItemHasKeyboardFocus,
      mInputManager->pSelectedGuiItem.get() &&
          mInputManager->pSelectedGuiItem.get()->getIsKeyboardInputElementFocused());
  updatePort(mHoveredGuiItemAllowsScrolling,
      mInputManager->pHoveredGuiItem.get() && mInputManager->pHoveredGuiItem.get()->getCanScroll());

  updatePort(mHoveredGuiItem, mInputManager->pHoveredGuiItem.get());
  updatePort(mActiveGuiItem, mInputManager->pActiveGuiItem.get());
  updatePort(mSelectedGuiItem, mInputManager->pSelectedGuiItem.get());

  updatePort(mHoveredNode, mInputManager->pHoveredNode.get());
  updatePort(mActiveNode, mInputManager->pActiveNode.get());
  updatePort(mSelectedNode, mInputManager->pSelectedNode.get());

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GetSelectionStateNodeCreate::GetSelectionStateNodeCreate(cs::core::InputManager* pInputManager)
    : mInputManager(pInputManager) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

IVdfnNode* GetSelectionStateNodeCreate::CreateNode(const VistaPropertyList& oParams) const {
  return new GetSelectionStateNode(mInputManager, oParams.GetSubListConstRef("param"));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
