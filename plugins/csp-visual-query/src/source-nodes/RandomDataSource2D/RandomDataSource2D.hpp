////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_RANDOMDATASOURCE_2D_HPP
#define CSP_VISUAL_QUERY_RANDOMDATASOURCE_2D_HPP

#include "../../../../csl-node-editor/src/Node.hpp"
#include "../../types/types.hpp"

#include <random>

namespace csp::visualquery {

class RandomDataSource2D final : public csl::nodeeditor::Node {
 public:
  static const std::string                   sName;
  static std::string                         sSource();
  static std::unique_ptr<RandomDataSource2D> sCreate();

   RandomDataSource2D() noexcept;
  ~RandomDataSource2D() noexcept override;

  const std::string& getName() const noexcept override;

  void process() noexcept override;

  void onMessageFromJS(nlohmann::json const& message) override;

  nlohmann::json getData() const override;
  void           setData(nlohmann::json const& json) override;

 private:
  std::shared_ptr<Image2D> mData;

  csl::ogc::Bounds2D mBounds;

  std::mt19937                     mRandomNumberGenerator;
  std::uniform_real_distribution<> mDistribution;
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_RANDOMDATASOURCE_2D_HPP
