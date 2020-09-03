////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GET_SELECTION_STATE_NODE_HPP
#define CS_GET_SELECTION_STATE_NODE_HPP

#include <VistaBase/VistaVectorMath.h>
#include <VistaDataFlowNet/VdfnNode.h>
#include <VistaDataFlowNet/VdfnNodeFactory.h>
#include <VistaDataFlowNet/VdfnPort.h>
#include <VistaDataFlowNet/VdfnSerializer.h>
#include <VistaKernel/VistaKernelConfig.h>

#include "../cs-utils/AnimatedValue.hpp"

#include <memory>

namespace cs::core {
class InputManager;
} // namespace cs::core

/// This DFN node can be used to check whether scenegraph nodes or user interface elemnts are
/// currently hovered, selected or active. Available outports are:
/// bool: "selected_gui_item_has_keyboard_focus"
/// bool: "hovered_gui_item_allows_scrolling"
/// bool: "hovered_gui_item"
/// bool: "active_gui_item"
/// bool: "selected_gui_item"
/// bool: "hovered_node"
/// bool: "active_node"
/// bool: "selected_node"
class GetSelectionStateNode : public IVdfnNode {
 public:
  GetSelectionStateNode(cs::core::InputManager* pInputManager, VistaPropertyList const& oParams);

 protected:
  bool DoEvalNode() override;

 private:
  cs::core::InputManager* mInputManager;

  TVdfnPort<bool>* mSelectedGuiItemHasKeyboardFocus;
  TVdfnPort<bool>* mHoveredGuiItemAllowsScrolling;

  TVdfnPort<bool>* mHoveredGuiItem;
  TVdfnPort<bool>* mActiveGuiItem;
  TVdfnPort<bool>* mSelectedGuiItem;

  TVdfnPort<bool>* mHoveredNode;
  TVdfnPort<bool>* mActiveNode;
  TVdfnPort<bool>* mSelectedNode;
};

class GetSelectionStateNodeCreate : public VdfnNodeFactory::IVdfnNodeCreator {
 public:
  GetSelectionStateNodeCreate(cs::core::InputManager* pInputManager);
  IVdfnNode* CreateNode(const VistaPropertyList& oParams) const override;

 private:
  cs::core::InputManager* mInputManager;
};

#endif // CS_GET_SELECTION_STATE_NODE_HPP
