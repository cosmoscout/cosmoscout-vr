////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_RANDOMDATASOURCE_HPP
#define CSP_VISUAL_QUERY_RANDOMDATASOURCE_HPP

#include "../../types/types.hpp"
#include <random>

namespace csp::visualquery {

class RandomDataSource final : public csl::nodeeditor::Node {
 public:
  static const std::string                 sName;
  static std::string                       sSource();
  static std::unique_ptr<RandomDataSource> sCreate();

  RandomDataSource() noexcept;
  ~RandomDataSource() noexcept override;

  const std::string& getName() const noexcept override;

  void process() noexcept override;

  void onMessageFromJS(nlohmann::json const& message) override;

  nlohmann::json getData() const override;
  void           setData(nlohmann::json const& json) override;

 private:
  std::shared_ptr<Image2D> mData;

  double mMinLat;
  double mMinLon;

  double mMaxLat;
  double mMaxLon;

  std::mt19937                           mRandomNumberGenerator;
  std::uniform_real_distribution<double> mDistribution;
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_RANDOMDATASOURCE_HPP
