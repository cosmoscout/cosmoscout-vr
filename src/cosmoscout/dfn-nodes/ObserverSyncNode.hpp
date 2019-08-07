////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef VIRTUAL_PLANET_OBSERVER_SYNC_NODE_HPP
#define VIRTUAL_PLANET_OBSERVER_SYNC_NODE_HPP

#include <VistaBase/VistaVectorMath.h>
#include <VistaDataFlowNet/VdfnNode.h>
#include <VistaDataFlowNet/VdfnNodeFactory.h>
#include <VistaDataFlowNet/VdfnPort.h>
#include <VistaDataFlowNet/VdfnSerializer.h>
#include <VistaKernel/EventManager/VistaEventHandler.h>
#include <VistaKernel/VistaKernelConfig.h>

#include <glm/gtx/quaternion.hpp>
#include <memory>

class VistaExternalMsgEvent;
class VistaMsg;

namespace cs::core {
class TimeControl;
class SolarSystem;
} // namespace cs::core

/// Node which synchronizes the observer position between all connected cluster nodes.
class ObserverSyncNode : public IVdfnNode, public VistaEventHandler {
 public:
  ObserverSyncNode(std::shared_ptr<cs::core::SolarSystem> const& pSolarSystem,
      std::shared_ptr<cs::core::TimeControl> const&              pTimeControl);
  ~ObserverSyncNode() override;

  bool PrepareEvaluationRun() override;

  void HandleEvent(VistaEvent* pEvent) override;

 protected:
  bool DoEvalNode() override;

 private:
  struct SyncMessage {
    glm::dvec3 position;
    glm::dquat rotation;
    double     scale;
    int        centerLength;
    int        frameLength;
    double     time;
  };

  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
  std::shared_ptr<cs::core::TimeControl> mTimeControl;

  TVdfnPort<double>* mTime;

  double mLastTime;

  VistaExternalMsgEvent* mEvent   = nullptr;
  VistaMsg*              mMessage = nullptr;
};

class ObserverSyncNodeCreate : public VdfnNodeFactory::IVdfnNodeCreator {
 public:
  ObserverSyncNodeCreate(std::shared_ptr<cs::core::SolarSystem> const& pSolarSystem,
      std::shared_ptr<cs::core::TimeControl> const&                    pTimeControl);
  IVdfnNode* CreateNode(const VistaPropertyList& oParams) const override;

 private:
  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
  std::shared_ptr<cs::core::TimeControl> mTimeControl;
};

#endif // VIRTUAL_PLANET_OBSERVER_SYNC_NODE_HPP
