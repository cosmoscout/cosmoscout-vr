////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_OVERLAYRENDER_HPP
#define CSP_VISUAL_QUERY_OVERLAYRENDER_HPP

#include "../../../../csl-node-editor/src/Node.hpp"
#include "Renderer.hpp"

namespace csp::visualquery {

class OverlayRender final : public csl::nodeeditor::Node {
 public:
  // static interface ------------------------------------------------------------------------------

  static const std::string              sName;
  static std::string                    sSource();
  static std::unique_ptr<OverlayRender> sCreate(std::shared_ptr<cs::core::SolarSystem> solarSystem,
      std::shared_ptr<cs::core::Settings>                                              settings);

  // instance interface ----------------------------------------------------------------------------

  OverlayRender(std::shared_ptr<cs::core::SolarSystem> solarSystem,
      std::shared_ptr<cs::core::Settings>              settings);
  ~OverlayRender() override;

  /// Each node must override this. It simply returns the static sName.
  std::string const& getName() const override;

  void init() override;

  /// Whenever the simulation time changes, the TimeNode will call this method itself. It simply
  /// updates the value of the 'time' output. This method may also get called occasionally by the
  /// node editor, for example if a new web client was connected hence needs updated values for all
  /// nodes.
  void           process() override;
  void           onMessageFromJS(const nlohmann::json& message) override;
  nlohmann::json getData() const override;
  void           setData(const nlohmann::json& json) override;

 private:
  std::unique_ptr<Renderer>        mRenderer;
  std::unique_ptr<VistaOpenGLNode> mGLNode;

  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
  std::shared_ptr<cs::core::Settings>    mSettings;
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_OVERLAYRENDER_HPP