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
#include <spdlog/spdlog.h>

namespace cs::core {

////////////////////////////////////////////////////////////////////////////////////////////////////

InputManager::InputManager() {

  // Tell the user what's going on.
  spdlog::debug("Creating InputManager.");

  mClickTime = boost::posix_time::microsec_clock::universal_time();

  auto pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

  // Configure selection volume.
  VistaEvenCone selectionVolume(10.f, 0.3f);
  mSelection.SetSelectionVolume(selectionVolume);
  mSelection.SetStickyness(0.5f);
  mSelection.SetSnappiness(0.5f);

  // Add selection ray to scenegraph.
  VistaGeometryFactory oGeometryFactory(pSG);

  VistaTransformNode* pIntentionNode =
      pSG->NewTransformNode(dynamic_cast<VistaGroupNode*>(pSG->GetNode("Virtualplatform-Node")));
  pIntentionNode->SetName("SELECTION_NODE");
  GetVistaSystem()->GetDfnObjectRegistry()->SetObject("SELECTION_NODE", nullptr, pIntentionNode);

  // Unbind q-action.
  GetVistaSystem()->GetKeyboardSystemControl()->UnbindAction('q');

  // Copy, paste, ctr-z,...
  GetVistaSystem()->GetKeyboardSystemControl()->BindAction(24, [this]() {
    if (pHoveredGuiNode.get()) {
      pHoveredGuiNode.get()->cut();
    }
  });
  GetVistaSystem()->GetKeyboardSystemControl()->BindAction(3, [this]() {
    if (pHoveredGuiNode.get()) {
      pHoveredGuiNode.get()->copy();
    }
  });
  GetVistaSystem()->GetKeyboardSystemControl()->BindAction(22, [this]() {
    if (pHoveredGuiNode.get()) {
      pHoveredGuiNode.get()->paste();
    }
  });
  GetVistaSystem()->GetKeyboardSystemControl()->BindAction(26, [this]() {
    if (pHoveredGuiNode.get()) {
      pHoveredGuiNode.get()->undo();
    }
  });
  GetVistaSystem()->GetKeyboardSystemControl()->BindAction(25, [this]() {
    if (pHoveredGuiNode.get()) {
      pHoveredGuiNode.get()->redo();
    }
  });

  // Register handler and events.
  GetVistaSystem()->GetKeyboardSystemControl()->SetDirectKeySink(this);
  GetVistaSystem()->GetEventManager()->AddEventHandler(
      this, VistaInteractionEvent::GetTypeId(), VistaInteractionEvent::VEID_GRAPH_INPORT_CHANGE);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

InputManager::~InputManager() {
  // Tell the user what's going on.
  spdlog::debug("Deleting InputManager.");

  for (auto adapter : mAdapters) {
    mSelection.UnregisterNode(adapter);
    delete adapter;
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
    auto pAdapter = new VistaBoundingBoxAdapter(pNode);
    mSelection.RegisterNode(pAdapter);
    mAdapters.insert(pAdapter);
  } else {
    auto pAdapter = new VistaNodeAdapter(pNode);
    mSelection.RegisterNode(pAdapter);
    mAdapters.insert(pAdapter);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void InputManager::unregisterSelectable(std::shared_ptr<utils::IntersectableObject> const& pBody) {
  if (pBody) {
    mIntersectables.erase(pBody);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void InputManager::unregisterSelectable(gui::ScreenSpaceGuiArea* pGui) {
  if (pGui) {
    mScreenSpaceGuis.erase(pGui);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void InputManager::unregisterSelectable(IVistaNode* pNode) {
  if (pNode) {
    for (auto& pAdapter : mAdapters) {
      if (pAdapter->GetNode() == pNode) {
        mSelection.UnregisterNode(pAdapter);
        delete pAdapter;
        mAdapters.erase(pAdapter);
        return;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void InputManager::update() {
  auto pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();

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
    auto           pProbs = pViewport->GetProjection()->GetProjectionProperties();

    VistaVector3D v3Origin, v3Up, v3Normal;
    double        dLeft, dRight, dBottom, dTop;

    pProbs->GetProjPlaneExtents(dLeft, dRight, dBottom, dTop);
    pProbs->GetProjPlaneMidpoint(v3Origin[0], v3Origin[1], v3Origin[2]);
    pProbs->GetProjPlaneUp(v3Up[0], v3Up[1], v3Up[2]);
    pProbs->GetProjPlaneNormal(v3Normal[0], v3Normal[1], v3Normal[2]);

    auto platformTransform =
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
        int x = (int)((guiIntersection[0] - dLeft) / (dRight - dLeft) * pViewportGui->getWidth());
        int y = (int)((1.0 - (guiIntersection[1] - dBottom) / (dTop - dBottom)) *
                      pViewportGui->getHeight());

        gui::GuiItem* item = pViewportGui->getItemAt(x, y);

        // If there is a pActiveGuiNode in this pViewportGui, we want to send the input events even
        // if the pointer is not over the item
        if (!item && pActiveGuiNode.get()) {
          if (std::find(pViewportGui->getItems().begin(), pViewportGui->getItems().end(),
                  pActiveGuiNode.get()) != pViewportGui->getItems().end()) {
            item = pActiveGuiNode.get();
          }
        }

        if (item) {
          // There has been no focused gui element before, so inject a focus event.
          if (!pHoveredGuiNode.get()) {
            item->injectFocusEvent(true);
          }

          if (!pActiveGuiNode.get()) {
            pHoveredGuiNode = item;
            pHoveredNode    = nullptr;
          }

          if (!pActiveGuiNode.get() || pActiveGuiNode.get() == item) {
            gui::MouseEvent event;
            item->calculateMousePosition(x, y, event.mX, event.mY);
            item->injectMouseEvent(event);
          }
          return;
        }
      }
    }
  }

  // If there is an active guiItem in Worldspace, we want to send the mouse input to it in all
  // cases. So we can skip any later intersection calculation.
  if (pActiveGuiNode.get()) {
    VistaTransformMatrix matTransform;
    mActiveWorldSpaceGuiNode->GetParentWorldTransform(matTransform);
    gui::WorldSpaceGuiArea* area =
        dynamic_cast<gui::WorldSpaceGuiArea*>(mActiveWorldSpaceGuiNode->GetExtension());

    // We scale the gui element's transform matrix so that the translation magnitude becomes 1.f.
    // This is not strictly necessary but reduces precision issues in the inversion of the matrix.
    // Without this step, objects which are *really* far away would not be selectable.
    VistaTransformMatrix matScale;
    float                scale = 1.f / matTransform.GetTranslation().GetLength();
    matScale.SetToScaleMatrix(scale);
    matTransform = matScale * matTransform;

    VistaTransformMatrix matInvParentTransform = matTransform.GetInverted();
    VistaVector3D        start                 = matInvParentTransform * (v3Position * scale);
    VistaVector3D end = matInvParentTransform * ((v3Position * scale) + qOrientation.GetViewDir());

    int x, y;
    area->calculateMousePosition(start, end, x, y);
    // Inject a mouse move event.
    gui::MouseEvent event;
    pHoveredGuiNode.get()->calculateMousePosition(x, y, event.mX, event.mY);
    pHoveredGuiNode.get()->injectMouseEvent(event);
    return;
  }

  // There is no currently active object. So we can intersect our selectables to find potential
  // candidates.
  std::vector<IVistaIntentionSelectAdapter*> vResults;
  mSelection.SetConeTransform(v3Position, qOrientation);
  mSelection.Update(vResults);

  auto unselectGuiNode([this]() {
    if (pHoveredGuiNode.get()) {
      gui::MouseEvent event;
      event.mType = gui::MouseEvent::Type::eLeave;
      pHoveredGuiNode.get()->injectMouseEvent(event);
      pHoveredGuiNode.get()->injectFocusEvent(false);
      pHoveredGuiNode = nullptr;
    }
  });

  for (auto pNode : vResults) {
    auto pNodeAdapter = dynamic_cast<VistaNodeAdapter*>(pNode);
    auto pOGLNode     = dynamic_cast<VistaOpenGLNode*>(pNodeAdapter->GetNode());

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
      float                scale = 1.f / matTransform.GetTranslation().GetLength();
      matScale.SetToScaleMatrix(scale);
      matTransform = matScale * matTransform;

      VistaTransformMatrix matInvParentTransform = matTransform.GetInverted();
      VistaVector3D        start                 = matInvParentTransform * (v3Position * scale);
      VistaVector3D        end =
          matInvParentTransform * ((v3Position * scale) + qOrientation.GetViewDir());

      int x, y;
      if (area->calculateMousePosition(start, end, x, y)) {
        gui::GuiItem* item = area->getItemAt(x, y);
        if (item) {
          // There has been no focused gui element before, so inject a focus event.
          if (!pHoveredGuiNode.get()) {
            item->injectFocusEvent(true);
          }

          pHoveredGuiNode          = item;
          mActiveWorldSpaceGuiNode = pOGLNode;

          // Inject a mouse move event.
          gui::MouseEvent event;
          pHoveredGuiNode.get()->calculateMousePosition(x, y, event.mX, event.mY);
          pHoveredGuiNode.get()->injectMouseEvent(event);
          pHoveredNode = nullptr;
          return;
        }
      }
    }

    // Do not change input if mouse button is pressed.
    if (pButtons[0].get()) {
      return;
    }

    // It's a gui node, but it's completely transparent, so look for the next object.
    if (area) {
      continue;
    }

    // It is definitely no OpenGLNode, so deselect any previous active element.
    unselectGuiNode();

    IVistaNode* pVistaNode = pNodeAdapter->GetNode();

    if (pVistaNode) {
      pHoveredNode = pVistaNode;
      return;
    }
  }

  pHoveredNode = nullptr;
  unselectGuiNode();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool InputManager::HandleKeyPress(int key, int mods, bool bIsKeyRepeat) {

  if (key == VISTA_KEY_ESC) {
    sOnEscapePressed.emit();
    return true;
  }

  if (!pHoveredGuiNode.get()) {
    // continue event propagation if no gui not is hovered
    return false;
  }

  pHoveredGuiNode.get()->injectKeyEvent(gui::KeyEvent(key, mods));

  if (key == 24 || key == 3 || key == 22 || key == 26 || key == 25) {
    // Continue event propagation for ctrl-x, -c, -v, -z, -y.
    return false;
  }

  if (key < 0) {
    // Continue propagation for key-up events so that DFN realizes those even with the pointer being
    // above the gui.
    return false;
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void InputManager::HandleEvent(VistaEvent* pEvent) {
  if (pEvent->GetId() == VistaInteractionEvent::VEID_GRAPH_INPORT_CHANGE) {
    auto             event(dynamic_cast<VistaInteractionEvent*>(pEvent));
    IVdfnNode const* node(event->GetEventNode());
    std::string      tag;
    if (node->GetUserTag(tag)) {
      for (int i(0); i < pButtons.size(); ++i) {
        if (tag == "button_0" + std::to_string(i + 1)) {
          auto port(dynamic_cast<TVdfnPort<bool>*>(node->GetInPort("value")));
          pButtons[i] = port->GetValue();

          if (i == 0) {
            if (pHoveredNode.get()) {
              if (port->GetValue()) {
                pSelectedNode = pHoveredNode;
                pActiveNode   = pHoveredNode;
              } else {
                pActiveNode = nullptr;
              }
            }

            if (pHoveredGuiNode.get()) {
              gui::MouseEvent mouseEvent;
              mouseEvent.mButton = gui::Button::eLeft;
              if (port->GetValue()) {
                pSelectedGuiNode = pHoveredGuiNode;
                pActiveGuiNode   = pHoveredGuiNode;
                mouseEvent.mType = gui::MouseEvent::Type::ePress;
              } else {
                pActiveGuiNode   = nullptr;
                mouseEvent.mType = gui::MouseEvent::Type::eRelease;
              }
              pHoveredGuiNode.get()->injectMouseEvent(mouseEvent);
            }

            if (!port->GetValue()) {
              auto t    = boost::posix_time::microsec_clock::universal_time();
              auto diff = (t - mClickTime);

              if (diff < boost::posix_time::time_duration(boost::posix_time::millisec(200))) {
                sOnDoubleClick.emit();
              }
              mClickTime = t;
            }
          }
        }
      }

      if (tag == "scroll_wheel") {
        TVdfnPort<int>* port(dynamic_cast<TVdfnPort<int>*>(node->GetInPort("value")));
        if (port && pHoveredGuiNode.get()) {
          gui::MouseEvent mouseEvent;
          mouseEvent.mType = gui::MouseEvent::Type::eScroll;
          mouseEvent.mY    = port->GetValue() * 20;
          pHoveredGuiNode.get()->injectMouseEvent(mouseEvent);
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::core
