////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_CORE_INPUT_MANAGER_HPP
#define CS_CORE_INPUT_MANAGER_HPP

#include "cs_core_export.hpp"

#include "../cs-utils/IntersectableObject.hpp"
#include "../cs-utils/Property.hpp"
#include "Settings.hpp"

#include <VistaKernel/InteractionManager/VistaIntentionSelect.h>
#include <VistaKernel/InteractionManager/VistaKeyboardSystemControl.h>
#include <VistaKernel/Stuff/VistaInteractionHandlerBase.h>

#include <glm/glm.hpp>
#include <memory>
#include <unordered_set>

#include <boost/date_time/posix_time/posix_time.hpp>

class IVistaNode;
class VistaTransformNode;
class VistaNodeAdapter;

namespace cs::gui {
class GuiItem;
class ScreenSpaceGuiArea;
} // namespace cs::gui

namespace cs::core {

/// The central access point for handling input. An instance of this class is passed to all plugins.
class CS_CORE_EXPORT InputManager : public VistaKeyboardSystemControl::IVistaDirectKeySink,
                                    public IVistaInteractionHandlerBase {
 public:
  /// This class describes an intersection point on an IntersectableObject.
  struct Intersection {
    std::shared_ptr<utils::IntersectableObject> mObject   = nullptr;
    glm::dvec3                                  mPosition = glm::dvec3(0.0, 0.0, 0.0);

    bool operator==(Intersection const& other) const {
      return mObject == other.mObject && mPosition == other.mPosition;
    }
    bool operator!=(Intersection const& other) const {
      return mObject != other.mObject || mPosition != other.mPosition;
    }
  };

  /// The node that was clicked on last.
  utils::Property<IVistaNode*> pSelectedNode = nullptr;

  /// DocTODO not sure what this is for.
  utils::Property<IVistaNode*> pHoveredNode = nullptr;

  /// The GuiItem the mouse is currently hovering over.
  utils::Property<gui::GuiItem*> pHoveredGuiNode = nullptr;

  /// The IntersectableObject currently hovered over.
  utils::Property<Intersection> pHoveredObject;

  std::array<utils::Property<bool>, 8> pButtons;

  /// Emits an event when the escape key is pressed.
  utils::Signal<> sOnEscapePressed;

  /// Emits an event when a double click occurred.
  utils::Signal<> sOnDoubleClick;

  explicit InputManager();
  ~InputManager() override;

  /// Register an IntersectableObject to be selectable.
  void registerSelectable(std::shared_ptr<utils::IntersectableObject> const& pBody);

  /// Register an IVistaNode to be selectable.
  void registerSelectable(IVistaNode* pNode);

  /// Register a static GUI element to be selectable.
  void registerSelectable(gui::ScreenSpaceGuiArea* pGui);

  /// Unregister an IntersectableObject for being selectable.
  void unregisterSelectable(std::shared_ptr<utils::IntersectableObject> const& pBody);
  void unregisterSelectable(IVistaNode* pNode);
  void unregisterSelectable(gui::ScreenSpaceGuiArea* pGui);

  /// VistaEventHandler interface implementation, for gui updates.
  void HandleEvent(VistaEvent* pEvent) override;

  /// IVistaDirectKeySink interface, for keyboard input into the ui.
  bool HandleKeyPress(int key, int mods, bool bIsKeyRepeat) override;

  /// IVistaInteractionHandlerBase interface, for mouse input.
  bool HandleContextChange(VistaInteractionEvent* pEvent) override;
  bool HandleGraphUpdate(VistaInteractionEvent* pEvent) override;
  bool HandleTimeUpdate(double dTs, double dLastTs) override;

 private:
  VistaIntentionSelect                                            mSelection;
  std::unordered_set<VistaNodeAdapter*>                           mAdapters;
  std::unordered_set<std::shared_ptr<utils::IntersectableObject>> mIntersectables;
  std::unordered_set<gui::ScreenSpaceGuiArea*>                    mScreenSpaceGuis;
  VistaTransformNode*                                             mRayTrans;
  boost::posix_time::ptime                                        mClickTime;
};

} // namespace cs::core

#endif // CS_CORE_INPUT_MANAGER_HPP
