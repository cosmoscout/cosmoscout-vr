////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ObserverSyncNode.hpp"

#include "../../cs-core/SolarSystem.hpp"
#include "../../cs-core/TimeControl.hpp"

#include <VistaAspects/VistaPropertyAwareable.h>
#include <VistaInterProcComm/Connections/VistaMsg.h>
#include <VistaKernel/Cluster/VistaClusterMode.h>
#include <VistaKernel/EventManager/VistaEventManager.h>
#include <VistaKernel/EventManager/VistaExternalMsgEvent.h>
#include <VistaKernel/VistaSystem.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

const int MESSAGE_TAG = 235667;

////////////////////////////////////////////////////////////////////////////////////////////////////

ObserverSyncNode::ObserverSyncNode(std::shared_ptr<cs::core::SolarSystem> const& pSolarSystem,
    std::shared_ptr<cs::core::TimeControl> const&                                pTimeControl)
    : IVdfnNode()
    , mSolarSystem(pSolarSystem)
    , mTimeControl(pTimeControl)
    , mTime(nullptr)
    , mLastTime(-1.0) {
  RegisterInPortPrototype("time", new TVdfnPortTypeCompare<TVdfnPort<double>>);

  // configure our event
  mEvent = new VistaExternalMsgEvent();
  mEvent->SetId(VistaExternalMsgEvent::VEID_INCOMING_MSG);

  // we set a ticket to identify the message
  mMessage = new VistaMsg();
  mMessage->SetMsgTicket(MESSAGE_TAG);
  mEvent->SetThisMsg(mMessage);

  VistaEventManager* pEventManager = GetVistaSystem()->GetEventManager();
  pEventManager->AddEventHandler(
      this, VistaExternalMsgEvent::GetTypeId(), VistaExternalMsgEvent::VEID_INCOMING_MSG);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ObserverSyncNode::~ObserverSyncNode() {
  VistaEventManager* pEventManager = GetVistaSystem()->GetEventManager();
  pEventManager->RemEventHandler(
      this, VistaExternalMsgEvent::GetTypeId(), VistaExternalMsgEvent::VEID_INCOMING_MSG);

  delete mEvent;
  delete mMessage;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ObserverSyncNode::PrepareEvaluationRun() {
  mTime = dynamic_cast<TVdfnPort<double>*>(GetInPort("time"));
  return GetIsValid();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ObserverSyncNode::HandleEvent(VistaEvent* pEvent) {
  if (pEvent->GetType() == VistaExternalMsgEvent::GetTypeId()) {
    auto pMsgEvent = dynamic_cast<VistaExternalMsgEvent*>(pEvent);
    if (pMsgEvent && pMsgEvent->GetThisMsg()->GetMsgTicket() == MESSAGE_TAG) {
      SyncMessage msg;
      std::memcpy(&msg, &pMsgEvent->GetThisMsg()->GetThisMsgConstRef()[0], sizeof(SyncMessage));

      std::string center(
          (const char*)&pMsgEvent->GetThisMsg()->GetThisMsgConstRef()[sizeof(SyncMessage)],
          (unsigned long)msg.centerLength);
      std::string frame((const char*)&pMsgEvent->GetThisMsg()
                            ->GetThisMsgConstRef()[sizeof(SyncMessage) + msg.centerLength],
          (unsigned long)msg.frameLength);

      auto& oObs = mSolarSystem->getObserver();
      oObs.setAnchorPosition(msg.position);
      oObs.setAnchorRotation(msg.rotation);
      oObs.setAnchorScale(msg.scale);
      oObs.setCenterName(center);
      oObs.setFrameName(frame);

      mTimeControl->pSimulationTime = msg.time;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool ObserverSyncNode::DoEvalNode() {
  double dTtime = mTime->GetValue();

  // skip first update since we do not know the delta time
  if (mLastTime < 0.0) {
    mLastTime = dTtime;
    return true;
  }

  // ensures that we don't loose time on many very small updates
  double dDeltaTime = dTtime - mLastTime;
  if (dDeltaTime < Vista::Epsilon) {
    return true;
  }

  mLastTime = dTtime;

  auto& oObs = mSolarSystem->getObserver();

  if (GetVistaSystem()->GetClusterMode()->GetIsLeader()) {
    SyncMessage msg;
    msg.position     = oObs.getAnchorPosition();
    msg.rotation     = oObs.getAnchorRotation();
    msg.scale        = oObs.getAnchorScale();
    msg.centerLength = (int)oObs.getCenterName().length();
    msg.frameLength  = (int)oObs.getFrameName().length();
    msg.time         = mTimeControl->pSimulationTime.get();

    int messageLength = sizeof(SyncMessage) + msg.centerLength + msg.frameLength;
    mMessage->GetThisMsgRef().resize((unsigned long)messageLength);

    std::memcpy(&mMessage->GetThisMsgRef()[0], &msg, sizeof(SyncMessage));
    std::memcpy(&mMessage->GetThisMsgRef()[sizeof(SyncMessage)], &oObs.getCenterName()[0],
        (unsigned long)msg.centerLength);
    std::memcpy(&mMessage->GetThisMsgRef()[sizeof(SyncMessage) + msg.centerLength],
        &oObs.getFrameName()[0], (unsigned long)msg.frameLength);

    GetVistaSystem()->GetEventManager()->ProcessEvent(mEvent);
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

ObserverSyncNodeCreate::ObserverSyncNodeCreate(
    std::shared_ptr<cs::core::SolarSystem> const& pSolarSystem,
    std::shared_ptr<cs::core::TimeControl> const& pTimeControl)
    : mSolarSystem(pSolarSystem)
    , mTimeControl(pTimeControl) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

IVdfnNode* ObserverSyncNodeCreate::CreateNode(const VistaPropertyList& oParams) const {
  return new ObserverSyncNode(mSolarSystem, mTimeControl);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
