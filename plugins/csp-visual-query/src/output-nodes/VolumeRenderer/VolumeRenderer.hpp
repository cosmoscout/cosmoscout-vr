////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_VOLUMERENDERER_HPP
#define CSP_VISUAL_QUERY_VOLUMERENDERER_HPP

#include "../../../../csl-node-editor/src/Node.hpp"
#include "../../types/types.hpp"

class VistaOpenGLNode;

namespace cs::core {
class Settings;
class SolarSystem;
} // namespace cs::core

namespace csp::visualquery {

class SinglePassRaycaster;

class VolumeRenderer final : public csl::nodeeditor::Node {
 public:
  // static interface ------------------------------------------------------------------------------

  static const std::string               sName;
  static std::string                     sSource();
  static std::unique_ptr<VolumeRenderer> sCreate(std::shared_ptr<cs::core::SolarSystem> solarSystem,
      std::shared_ptr<cs::core::Settings>                                               settings);

  // instance interface ----------------------------------------------------------------------------

   VolumeRenderer(std::shared_ptr<cs::core::SolarSystem> solarSystem,
       std::shared_ptr<cs::core::Settings>               settings);
  ~VolumeRenderer() override;

  std::string const& getName() const override;

  void init() override;

  void           process() override;
  void           onMessageFromJS(const nlohmann::json& message) override;
  nlohmann::json getData() const override;
  void           setData(const nlohmann::json& json) override;

 private:
  std::unique_ptr<SinglePassRaycaster> mRenderer;
  std::unique_ptr<VistaOpenGLNode>     mGLNode;

  std::shared_ptr<Volume3D> mVolume;

  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
  std::shared_ptr<cs::core::Settings>    mSettings;
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_VOLUMERENDERER_HPP
