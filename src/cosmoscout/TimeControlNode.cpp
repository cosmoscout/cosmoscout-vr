////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "TimeControlNode.hpp"

#include "../cs-core/Settings.hpp"
#include "../cs-utils/convert.hpp"

#include <VistaAspects/VistaPropertyAwareable.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

TimeControlNode::TimeControlNode(
    cs::core::Settings* pSettings, VistaPropertyList const& oParams)
    : mSettings(pSettings)
    , mTime(nullptr)
    , mSimSpeed(nullptr)
    , mLastTime(-1.0) {

  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory): deleted in IVdfnNode::~IVdfnNode()
  RegisterInPortPrototype("time", new TVdfnPortTypeCompare<TVdfnPort<double>>);

  // NOLINTNEXTLINE(cppcoreguidelines-owning-memory): deleted in IVdfnNode::~IVdfnNode()
  RegisterInPortPrototype("sim_speed", new TVdfnPortTypeCompare<TVdfnPort<float>>);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TimeControlNode::PrepareEvaluationRun() {
  mTime        = dynamic_cast<TVdfnPort<double>*>(GetInPort("time"));
  mSimSpeed    = dynamic_cast<TVdfnPort<float>*>(GetInPort("sim_speed"));
  return GetIsValid();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TimeControlNode::GetIsValid() const {
  return (mSimSpeed && mTime);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TimeControlNode::DoEvalNode() {
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

  if (mSimSpeed) {
    float simSpeed = mSimSpeed->GetValue();
    mSettings->pTimeSpeed = simSpeed;
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TimeControlNodeCreate::TimeControlNodeCreate(cs::core::Settings* pSettings)
    : mSettings(pSettings) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

IVdfnNode* TimeControlNodeCreate::CreateNode(const VistaPropertyList& oParams) const {
  return new TimeControlNode(mSettings, oParams.GetSubListConstRef("param"));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
