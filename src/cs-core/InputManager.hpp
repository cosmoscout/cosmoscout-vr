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

#include <VistaKernel/EventManager/VistaEventHandler.h>
#include <VistaKernel/InteractionManager/VistaIntentionSelect.h>
#include <VistaKernel/InteractionManager/VistaKeyboardSystemControl.h>

#include <glm/glm.hpp>
#include <memory>
#include <unordered_set>

#include <boost/date_time/posix_time/posix_time.hpp>

class IVistaNode;
class VistaTransformNode;
class VistaNodeAdapter;
class VistaOpenGLNode;

namespace cs::gui {
class GuiItem;
class ScreenSpaceGuiArea;
} // namespace cs::gui

namespace cs::core {

/// The central access point for handling input. An instance of this class is passed to all plugins.
class CS_CORE_EXPORT InputManager : public VistaKeyboardSystemControl::IVistaDirectKeySink,
                                    public VistaEventHandler {
 public:
  /// This class describes an intersection point on an IntersectableObject. Usually this is used for
  /// intersections between the mouse ray and planets or moons.
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

  /// The GuiItem the pointer is currently hovering over is the pHoveredGuiNode. When pButtons[0] is
  /// pressed while hovering a GuiItem, this GuiItem will become active and selected. Once the
  /// button is released again, pActiveGuiNode will be nullptr again. However, it will stay selected
  /// until another GuiItem becomes selected. As long as a node is active, it will stay hovered and
  /// receive input events even if the pointer is actually hovering the item. These properties
  /// should be considered read-only.
  utils::Property<gui::GuiItem*> pHoveredGuiNode  = nullptr;
  utils::Property<gui::GuiItem*> pActiveGuiNode   = nullptr;
  utils::Property<gui::GuiItem*> pSelectedGuiNode = nullptr;

  /// When no GuiItem is hovered, any other IVistaNode which has been registered via
  /// registerSelectable() may become hovered.
  /// The IVistaNode the pointer is currently hovering over is the pHoveredNode. When pButtons[0] is
  /// pressed while hovering a registered IVistaNode, this IVistaNode will become active and
  /// selected. Once the button is released again, pActiveNode will be nullptr again. However, it
  /// will stay selected until another registered IVistaNode becomes selected. As long as a node is
  /// active, it will stay hovered and receive input events even if the pointer is actually hovering
  /// the item. These properties should be considered read-only.
  utils::Property<IVistaNode*> pHoveredNode  = nullptr;
  utils::Property<IVistaNode*> pActiveNode   = nullptr;
  utils::Property<IVistaNode*> pSelectedNode = nullptr;

  /// Regardless of the node state above, this property will always be updated. This property should
  /// be considered read-only.
  utils::Property<Intersection> pHoveredObject;

  /// Contains the state of the buttons of your input device. These properties should be considered
  /// read-only.
  std::array<utils::Property<bool>, 8> pButtons;

  /// Emits an event when the escape key is pressed. This signal should be considered read-only.
  utils::Signal<> sOnEscapePressed;

  /// Emits an event when a double click occurred. This signal should be considered read-only.
  utils::Signal<> sOnDoubleClick;

  /// Creates a new instance of this class. As a user, you will not need to call this directly, as
  /// an instance is created by CosmoScout's Application class.
  explicit InputManager();
  ~InputManager() override;

  /// Register an object to be selectable. You can register different types:
  ///  * utils::IntersectableObject: This is mainly used for cs::scene::CelestialBody. When
  ///      intersected, the IntersectableObject will be the pHoveredObject.
  ///  * IVistaNode: Intersections will be calculated via the VistaBoundingBoxAdapter. That means,
  ///      the GetBoundBox() method of the IVistaNode has to return the bounds which should be
  ///      checked for intersections.  If intersected, these nodes will be available in
  ///      pHoveredNode, pActiveNode and pSelectedNode. This type is also used for OpenGLNodes
  ///      containing gui::WorldSpaceGuiAreas. If such a node is intersected, the relevant GuiItem
  ///      will be available in pHoveredGuiNode, pActiveGuiNode and pSelectedGuiNode.
  ///  * gui::ScreenSpaceGuiArea: If such an object is intersected, it will be available in
  ///      pHoveredGuiNode, pActiveGuiNode and pSelectedGuiNode.
  void registerSelectable(std::shared_ptr<utils::IntersectableObject> const& pBody);
  void registerSelectable(IVistaNode* pNode);
  void registerSelectable(gui::ScreenSpaceGuiArea* pGui);

  /// Unregister any previously registered object. It's important to call these when the registered
  /// objects are deleted.
  void unregisterSelectable(std::shared_ptr<utils::IntersectableObject> const& pBody);
  void unregisterSelectable(IVistaNode* pNode);
  void unregisterSelectable(gui::ScreenSpaceGuiArea* pGui);

  /// This method computes the intersection between the mouse ray (SELECTION_NODE) and all
  /// registered objects.
  void update();

  // overrides of ViSTA base classes ---------------------------------------------------------------

  /// This is used to handle events emitted from DFN networks. That is button press and scroll wheel
  /// events. These are directly injected to the pHoveredGuiNode.
  void HandleEvent(VistaEvent* pEvent) override;

  /// This is used to inject key presses to the pHoveredGuiNode. The ESC key is handled differently,
  /// it will be used to fire the sOnEscapePressed signal.
  bool HandleKeyPress(int key, int mods, bool bIsKeyRepeat) override;

 private:
  VistaIntentionSelect                                            mSelection;
  std::unordered_set<VistaNodeAdapter*>                           mAdapters;
  std::unordered_set<std::shared_ptr<utils::IntersectableObject>> mIntersectables;
  std::unordered_set<gui::ScreenSpaceGuiArea*>                    mScreenSpaceGuis;
  VistaTransformNode*                                             mRayTrans;
  boost::posix_time::ptime                                        mClickTime;

  VistaOpenGLNode* mActiveWorldSpaceGuiNode;
};

} // namespace cs::core

#endif // CS_CORE_INPUT_MANAGER_HPP
