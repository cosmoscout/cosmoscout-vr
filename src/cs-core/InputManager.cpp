////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "InputManager.hpp"

#include "../cs-gui/GuiItem.hpp"
#include "../cs-gui/ScreenSpaceGuiArea.hpp"
#include "../cs-gui/WorldSpaceGuiArea.hpp"
#include "../cs-utils/utils.hpp"
#include "GuiManager.hpp"
#include "logger.hpp"

#include <VistaDataFlowNet/VdfnNode.h>
#include <VistaDataFlowNet/VdfnObjectRegistry.h>
#include <VistaDataFlowNet/VdfnPort.h>
#include <VistaKernel/DisplayManager/GlutWindowImp/VistaGlutWindowingToolkit.h>
#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaDisplaySystem.h>
#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/DisplayManager/VistaVirtualPlatform.h>
#include <VistaKernel/EventManager/VistaEventManager.h>
#include <VistaKernel/GraphicsManager/VistaGeomNode.h>
#include <VistaKernel/GraphicsManager/VistaGeometryFactory.h>
#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/InteractionManager/VistaInteractionEvent.h>
#include <VistaKernel/InteractionManager/VistaInteractionManager.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper function that converts the character on the keyboard to an int expected by ViSTA.
constexpr int vistaKeyCode(char c) {
  return c - 'a' + 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

InputManager::InputManager() {

  // Tell the user what's going on.
  logger().debug("Creating InputManager.");

  mClickTime = boost::posix_time::microsec_clock::universal_time();

  auto* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  float const selectionHeight = 10.0F;
  float const selectionRadius = 0.3F;
  // Configure selection volume.
  VistaEvenCone selectionVolume(selectionHeight, selectionRadius);
  mSelection.SetSelectionVolume(selectionVolume);
  mSelection.SetStickyness(0.5F);
  mSelection.SetSnappiness(0.5F);

  // Add selection ray to scenegraph.
  VistaGeometryFactory oGeometryFactory(pSG);

  VistaTransformNode* pIntentionNode =
      pSG->NewTransformNode(dynamic_cast<VistaGroupNode*>(pSG->GetNode("Virtualplatform-Node")));
  pIntentionNode->SetName("SELECTION_NODE");
  GetVistaSystem()->GetDfnObjectRegistry()->SetObject("SELECTION_NODE", nullptr, pIntentionNode);

  // Unbind q-action.
  GetVistaSystem()->GetKeyboardSystemControl()->UnbindAction('q');

  // Copy, paste, ctr-z,...

  GetVistaSystem()->GetKeyboardSystemControl()->BindAction(vistaKeyCode('x'), [this]() {
    if (pSelectedGuiItem.get()) {
      pSelectedGuiItem.get()->cut();
    }
  });

  GetVistaSystem()->GetKeyboardSystemControl()->BindAction(vistaKeyCode('c'), [this]() {
    if (pSelectedGuiItem.get()) {
      pSelectedGuiItem.get()->copy();
    }
  });

  GetVistaSystem()->GetKeyboardSystemControl()->BindAction(vistaKeyCode('v'), [this]() {
    if (pSelectedGuiItem.get()) {
      pSelectedGuiItem.get()->paste();
    }
  });

  GetVistaSystem()->GetKeyboardSystemControl()->BindAction(vistaKeyCode('z'), [this]() {
    if (pSelectedGuiItem.get()) {
      pSelectedGuiItem.get()->undo();
    }
  });

  GetVistaSystem()->GetKeyboardSystemControl()->BindAction(vistaKeyCode('y'), [this]() {
    if (pSelectedGuiItem.get()) {
      pSelectedGuiItem.get()->redo();
    }
  });

  // Register handler and events.
  GetVistaSystem()->GetKeyboardSystemControl()->SetDirectKeySink(this);
  GetVistaSystem()->GetEventManager()->AddEventHandler(
      this, VistaInteractionEvent::GetTypeId(), VistaInteractionEvent::VEID_GRAPH_INPORT_CHANGE);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

InputManager::~InputManager() {
  try {
    // Tell the user what's going on.
    logger().debug("Deleting InputManager.");
  } catch (...) {}

  for (auto* adapter : mAdapters) {
    mSelection.UnregisterNode(adapter);
    delete adapter; // NOLINT(cppcoreguidelines-owning-memory): unordered_set doesn't like smart
                    // pointers.
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void InputManager::registerSelectable(std::shared_ptr<utils::IntersectableObject> const& pBody) {
  if (pBody) {
    mIntersectables.insert(pBody);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void InputManager::registerSelectable(gui::ScreenSpaceGuiArea* pGui) {
  if (pGui) {
    mScreenSpaceGuis.insert(pGui);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void InputManager::registerSelectable(IVistaNode* pNode) {
  if ((pNode->GetType() == VISTA_GEOMNODE || pNode->GetType() == VISTA_OPENGLNODE)) {
    auto* pAdapter = new VistaBoundingBoxAdapter(pNode); // NOLINT(cppcoreguidelines-owning-memory):
                                                         // std::set doesn't like smart pointers.
    mSelection.RegisterNode(pAdapter);
    mAdapters.insert(pAdapter);
  } else {
    auto* pAdapter = new VistaNodeAdapter(pNode); // NOLINT(cppcoreguidelines-owning-memory):
                                                  // std::set doesn't like smart pointers.
    mSelection.RegisterNode(pAdapter);
    mAdapters.insert(pAdapter);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void InputManager::unregisterSelectable(std::shared_ptr<utils::IntersectableObject> const& pBody) {
  if (pBody) {
    mIntersectables.erase(pBody);

    if (pHoveredObject.get().mObject == pBody) {
      pHoveredObject = Intersection();
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void InputManager::unregisterSelectable(gui::ScreenSpaceGuiArea* pGui) {
  if (pGui) {
    if (utils::contains(pGui->getItems(), pHoveredGuiItem.get())) {
      pHoveredGuiItem = nullptr;
    }
    if (utils::contains(pGui->getItems(), pSelectedGuiItem.get())) {
      pSelectedGuiItem = nullptr;
    }
    if (utils::contains(pGui->getItems(), pActiveGuiItem.get())) {
      pActiveGuiItem = nullptr;
    }

    mScreenSpaceGuis.erase(pGui);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void InputManager::unregisterSelectable(IVistaNode* pNode) {
  if (pHoveredNode.get() == pNode) {
    pHoveredNode = nullptr;
  }
  if (pSelectedNode.get() == pNode) {
    pSelectedNode = nullptr;
  }
  if (pActiveNode.get() == pNode) {
    pActiveNode = nullptr;
  }

  auto* pOGLNode = dynamic_cast<VistaOpenGLNode*>(pNode);

  // If its an OpenGLNode, it may be a gui element.
  if (pOGLNode) {
    auto* area = dynamic_cast<gui::WorldSpaceGuiArea*>(pOGLNode->GetExtension());
    if (area) {
      if (utils::contains(area->getItems(), pHoveredGuiItem.get())) {
        pHoveredGuiItem = nullptr;
      }
      if (utils::contains(area->getItems(), pSelectedGuiItem.get())) {
        pSelectedGuiItem = nullptr;
      }
      if (utils::contains(area->getItems(), pActiveGuiItem.get())) {
        pActiveGuiItem = nullptr;
      }
    }
  }

  if (pNode) {
    for (auto* pAdapter : mAdapters) {
      if (pAdapter->GetNode() == pNode) {
        mSelection.UnregisterNode(pAdapter);
        delete pAdapter; // NOLINT(cppcoreguidelines-owning-memory): unordered_set doesn't like
                         // smart pointers.
        mAdapters.erase(pAdapter);
        return;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void InputManager::update() {
  auto* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  // Set position and orientation of selection ray.
  VistaVector3D       v3Position;
  VistaQuaternion     qOrientation;
  VistaTransformNode* pIntentionNode =
      dynamic_cast<VistaTransformNode*>(pSG->GetNode("SELECTION_NODE"));

  pIntentionNode->GetWorldPosition(v3Position);
  pIntentionNode->GetWorldOrientation(qOrientation);
  VistaVector3D v3Direction = qOrientation.GetViewDir();

  // Test the Intention Node for Intersection with planets.
  Intersection intersection;
  for (auto const& intersectable : mIntersectables) {
    glm::dvec3 pos(0.0, 0.0, 0.0);

    bool intersects =
        intersectable->getIntersection(glm::dvec3(v3Position[0], v3Position[1], v3Position[2]),
            glm::dvec3(v3Direction[0], v3Direction[1], v3Direction[2]), pos);

    if (intersects) {
      intersection.mObject   = intersectable;
      intersection.mPosition = pos;
      break;
    }
  }

  pHoveredObject = intersection;

  // If there is an active node, we do not want to change any selection state.
  if (pActiveNode.get()) {
    return;
  }

  // Test the Intention Node for Intersection with screen space gui.
  if (!mScreenSpaceGuis.empty()) {
    VistaViewport* pViewport(GetVistaSystem()->GetDisplayManager()->GetViewports().begin()->second);
    auto*          pProbs = pViewport->GetProjection()->GetProjectionProperties();

    VistaVector3D v3Origin;
    VistaVector3D v3Up;
    VistaVector3D v3Normal;
    double        dLeft{};
    double        dRight{};
    double        dBottom{};
    double        dTop{};

    pProbs->GetProjPlaneExtents(dLeft, dRight, dBottom, dTop);
    pProbs->GetProjPlaneMidpoint(v3Origin[0], v3Origin[1], v3Origin[2]);
    pProbs->GetProjPlaneUp(v3Up[0], v3Up[1], v3Up[2]);
    pProbs->GetProjPlaneNormal(v3Normal[0], v3Normal[1], v3Normal[2]);

    auto* platformTransform =
        GetVistaSystem()->GetDisplayManager()->GetDisplaySystem()->GetReferenceFrame();

    VistaVector3D start = v3Position;
    VistaVector3D end   = (v3Position + qOrientation.GetViewDir());

    start = platformTransform->TransformPositionToFrame(start);
    end   = platformTransform->TransformPositionToFrame(end);

    VistaVector3D direction(end - start);

    VistaRay   ray(start, direction);
    VistaPlane plane(v3Origin, v3Up.Cross(v3Normal), v3Up, v3Normal);

    VistaVector3D guiIntersection;

    if (plane.CalcIntersection(ray, guiIntersection)) {
      for (auto const& pViewportGui : mScreenSpaceGuis) {
        int x = static_cast<int>(
            (guiIntersection[0] - dLeft) / (dRight - dLeft) * pViewportGui->getWidth());
        int y = static_cast<int>(
            (1.0 - (guiIntersection[1] - dBottom) / (dTop - dBottom)) * pViewportGui->getHeight());

        if (pActiveGuiItem.get()) {
          // If there is an active gui item we need to send mouse move events even if it's not
          // hovered. So we check whether the current screen space gui area contains this active
          // item.
          if (utils::contains(pViewportGui->getItems(), pActiveGuiItem.get())) {

            // If so, we send the mouse move and are done.
            gui::MouseEvent event;
            pActiveGuiItem.get()->calculateMousePosition(x, y, event.mX, event.mY);
            pActiveGuiItem.get()->injectMouseEvent(event);
            return;
          }
        } else {
          // If there is no active gui item, we should check if the mouse pointer is currently
          // hovering an item.
          auto* item = pViewportGui->getItemAt(x, y);
          if (item) {
            // We have hovered gui item! Let's update the selection state as depicted in the diagram
            // in InputManager.hpp.
            pHoveredGuiItem = item;
            pHoveredNode    = nullptr;

            gui::MouseEvent event;
            item->calculateMousePosition(x, y, event.mX, event.mY);
            item->injectMouseEvent(event);
            return;
          }
        }
      }
    }
  }

  // Now we are sure that there is no active gui item of a screen space gui area. If there is an
  // active gui item in a Worldspace gui area, we want to send the mouse input to it in all cases.
  // So we can skip any later intersection calculation.
  if (pActiveGuiItem.get()) {
    VistaTransformMatrix matTransform;
    mActiveWorldSpaceGuiNode->GetParentWorldTransform(matTransform);
    auto* area = dynamic_cast<gui::WorldSpaceGuiArea*>(mActiveWorldSpaceGuiNode->GetExtension());

    // We scale the gui element's transform matrix so that the translation magnitude becomes 1.f.
    // This is not strictly necessary but reduces precision issues in the inversion of the matrix.
    // Without this step, objects which are *really* far away would not be selectable.
    VistaTransformMatrix matScale;
    float                scale = 1.F / matTransform.GetTranslation().GetLength();
    matScale.SetToScaleMatrix(scale);
    matTransform = matScale * matTransform;

    VistaTransformMatrix matInvParentTransform = matTransform.GetInverted();
    VistaVector3D        start                 = matInvParentTransform * (v3Position * scale);
    VistaVector3D end = matInvParentTransform * ((v3Position * scale) + qOrientation.GetViewDir());

    int x{};
    int y{};
    area->calculateMousePosition(start, end, x, y);
    // Inject a mouse move event.
    gui::MouseEvent event;
    pActiveGuiItem.get()->calculateMousePosition(x, y, event.mX, event.mY);
    pActiveGuiItem.get()->injectMouseEvent(event);
    return;
  }

  // There is no currently active object. So we can intersect our selectables to find potential
  // candidates for hovering.
  std::vector<IVistaIntentionSelectAdapter*> vResults;
  mSelection.SetConeTransform(v3Position, qOrientation);
  mSelection.Update(vResults);

  for (auto* pNode : vResults) {
    auto* pNodeAdapter = dynamic_cast<VistaNodeAdapter*>(pNode);
    auto* pOGLNode     = dynamic_cast<VistaOpenGLNode*>(pNodeAdapter->GetNode());

    // If its an OpenGLNode, it may be a gui element.
    gui::WorldSpaceGuiArea* area =
        pOGLNode ? dynamic_cast<gui::WorldSpaceGuiArea*>(pOGLNode->GetExtension()) : nullptr;

    if (area) {
      VistaTransformMatrix matTransform;
      pOGLNode->GetParentWorldTransform(matTransform);

      // We scale the gui element's transform matrix so that the translation magnitude
      // becomes 1.f. This is not strictly necessary but reduces precision issues in the
      // inversion of the matrix. Without this step, objects which are *really* far away would
      // not be selectable.
      VistaTransformMatrix matScale;
      float                scale = 1.F / matTransform.GetTranslation().GetLength();
      matScale.SetToScaleMatrix(scale);
      matTransform = matScale * matTransform;

      VistaTransformMatrix matInvParentTransform = matTransform.GetInverted();
      VistaVector3D        start                 = matInvParentTransform * (v3Position * scale);
      VistaVector3D        end =
          matInvParentTransform * ((v3Position * scale) + qOrientation.GetViewDir());

      int x{};
      int y{};
      if (area->calculateMousePosition(start, end, x, y)) {
        gui::GuiItem* item = area->getItemAt(x, y);

        // We found an intersected gui item! Let's inject a mouse move event and we are done.
        if (item) {
          pHoveredGuiItem          = item;
          pHoveredNode             = nullptr;
          mActiveWorldSpaceGuiNode = pOGLNode;

          gui::MouseEvent event;
          pHoveredGuiItem.get()->calculateMousePosition(x, y, event.mX, event.mY);
          pHoveredGuiItem.get()->injectMouseEvent(event);
          return;
        }
      }

      // It's a gui node, but it's completely transparent, so look for the next object.
      continue;
    }

    // It is definitely no gui item, so deselect any previous active element.
    IVistaNode* pVistaNode = pNodeAdapter->GetNode();

    if (pVistaNode) {
      pHoveredNode = pVistaNode;

      // De-hover any hovered gui items.
      if (pHoveredGuiItem.get()) {
        gui::MouseEvent event;
        event.mType = gui::MouseEvent::Type::eLeave;
        pHoveredGuiItem.get()->injectMouseEvent(event);
        pHoveredGuiItem = nullptr;
        GuiManager::setCursor(gui::Cursor::ePointer);
      }

      return;
    }
  }

  // It seems nothing is hovered at all...
  if (pHoveredGuiItem.get()) {
    gui::MouseEvent event;
    event.mType = gui::MouseEvent::Type::eLeave;
    pHoveredGuiItem.get()->injectMouseEvent(event);
    pHoveredGuiItem = nullptr;
    GuiManager::setCursor(gui::Cursor::ePointer);
  }

  pHoveredNode = nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool InputManager::HandleKeyPress(int key, int mods, bool /*bIsKeyRepeat*/) {
  if (key == VISTA_KEY_ESC) {
    sOnEscapePressed.emit();
    return true;
  }

  if (pSelectedGuiItem.get() && pSelectedGuiItem.get()->getIsKeyboardInputElementFocused()) {
    pSelectedGuiItem.get()->injectKeyEvent(gui::KeyEvent(key, mods));

    // Continue propagation for key-up events so that DFN realizes those even with the pointer being
    // above the gui. Also, continue event propagation for ctrl-x, -c, -v, -z, -y.
    return !(key < 0 || key == vistaKeyCode('x') || key == vistaKeyCode('c') ||
             key == vistaKeyCode('v') || key == vistaKeyCode('z') || key == vistaKeyCode('y'));
  }

  // Continue event propagation to DFN for navigation input if no gui input element has focus.
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This implements the state transitions depicted in the diagram in InputManager.hpp which are
// labelled with "button press" and "button release".
template <typename T>
void handleButtonEvent(bool pressed, T& hovered, T& active, T& selected) {
  if (pressed) {
    if (hovered.get()) {
      active   = hovered;
      selected = hovered;
    } else if (selected.get()) {
      selected = nullptr;
    }
  } else {
    if (active.get()) {
      active = nullptr;
    }
  }
}

void InputManager::HandleEvent(VistaEvent* pEvent) {
  if (pEvent->GetId() == VistaInteractionEvent::VEID_GRAPH_INPORT_CHANGE) {
    auto*            event(dynamic_cast<VistaInteractionEvent*>(pEvent));
    IVdfnNode const* node(event->GetEventNode());
    std::string      tag;
    if (node->GetUserTag(tag)) {
      for (size_t i(0); i < pButtons.size(); ++i) {
        if (tag == "button_0" + std::to_string(i + 1)) {
          auto* port(dynamic_cast<TVdfnPort<bool>*>(node->GetInPort("value")));
          pButtons.at(i) = port->GetValue();

          if (i == 0) {

            // If the button is released and we have an active gui item, its active state will be
            // reset. So we send a mouse release event before.
            if (!port->GetValue() && pActiveGuiItem.get()) {
              gui::MouseEvent mouseEvent;
              mouseEvent.mButton = gui::Button::eLeft;
              mouseEvent.mType   = gui::MouseEvent::Type::eRelease;
              pActiveGuiItem.get()->injectMouseEvent(mouseEvent);
            }

            // If we have a button press event for a selected but not hovered item, it will be
            // un-selected. We should send focus-out event.
            if (port->GetValue() && pSelectedGuiItem.get() &&
                pSelectedGuiItem.get() != pHoveredGuiItem()) {

              pSelectedGuiItem.get()->injectFocusEvent(false);
            }

            // If there is a hovered item which is not yet selected, we should send a focus-in
            // event.
            if (port->GetValue() && pHoveredGuiItem.get() &&
                pSelectedGuiItem.get() != pHoveredGuiItem()) {

              pHoveredGuiItem.get()->injectFocusEvent(true);
            }

            // Execute the state transitions as depicted in the diagram in InputManager.hpp. All
            // transitions which are labelled with "button press" and "button release".
            handleButtonEvent(port->GetValue(), pHoveredGuiItem, pActiveGuiItem, pSelectedGuiItem);
            handleButtonEvent(port->GetValue(), pHoveredNode, pActiveNode, pSelectedNode);

            // Now inject the button press event.
            if (port->GetValue()) {
              gui::MouseEvent mouseEvent;
              mouseEvent.mButton = gui::Button::eLeft;
              mouseEvent.mType   = gui::MouseEvent::Type::ePress;

              if (pActiveGuiItem.get()) {
                pActiveGuiItem.get()->injectMouseEvent(mouseEvent);
              } else if (pHoveredGuiItem.get()) {
                pHoveredGuiItem.get()->injectMouseEvent(mouseEvent);
              }
            }

            if (!port->GetValue()) {
              auto t    = boost::posix_time::microsec_clock::universal_time();
              auto diff = (t - mClickTime);

              int32_t const doubleClickTimeMillis = 200;
              if (diff < boost::posix_time::time_duration(
                             boost::posix_time::millisec(doubleClickTimeMillis))) {
                sOnDoubleClick.emit();
              }
              mClickTime = t;
            }
          }
        }
      }

      if (tag == "scroll_wheel") {
        TVdfnPort<int>* port(dynamic_cast<TVdfnPort<int>*>(node->GetInPort("value")));
        auto*           item = pActiveGuiItem.get() ? pActiveGuiItem.get() : pHoveredGuiItem.get();
        if (port && item && item->getCanScroll()) {
          gui::MouseEvent mouseEvent;
          mouseEvent.mType = gui::MouseEvent::Type::eScroll;
          mouseEvent.mY    = port->GetValue() * 20;
          item->injectMouseEvent(mouseEvent);
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
