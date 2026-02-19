////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_TIME_CONTROL_NODE_HPP
#define CS_TIME_CONTROL_NODE_HPP

#include <VistaDataFlowNet/VdfnNode.h>
#include <VistaDataFlowNet/VdfnNodeFactory.h>
#include <VistaDataFlowNet/VdfnPort.h>
#include <VistaDataFlowNet/VdfnSerializer.h>
#include <VistaKernel/VistaKernelConfig.h>

#include <memory>

namespace cs::core {
class Settings;
} // namespace cs::core

class TimeControlNode : public IVdfnNode {
 public:
  TimeControlNode(cs::core::Settings* pSettings, VistaPropertyList const& oParams);

  bool PrepareEvaluationRun() override;

 protected:
  bool DoEvalNode() override;
  bool GetIsValid() const override;

 private:
  cs::core::Settings* mSettings;

  TVdfnPort<double>* mTime;
  TVdfnPort<float>*  mSimSpeed;

  double mLastTime;
};

class TimeControlNodeCreate : public VdfnNodeFactory::IVdfnNodeCreator {
 public:
  TimeControlNodeCreate(cs::core::Settings* pSettings);
  IVdfnNode* CreateNode(const VistaPropertyList& oParams) const override;

 private:
  cs::core::Settings* mSettings;
};

#endif // CS_TIME_CONTROL_NODE_HPP
