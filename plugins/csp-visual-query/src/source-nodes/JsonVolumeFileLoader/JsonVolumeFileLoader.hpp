////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_JSON_VOLUME_FILE_LOADER_HPP
#define CSP_VISUAL_QUERY_JSON_VOLUME_FILE_LOADER_HPP

#include "../../../../csl-node-editor/src/Node.hpp"
#include "../../types/types.hpp"

#include <random>

namespace csp::visualquery {

class JsonVolumeFileLoader final : public csl::nodeeditor::Node {
 public:
  static const std::string                     sName;
  static std::string                           sSource();
  static std::unique_ptr<JsonVolumeFileLoader> sCreate();

   JsonVolumeFileLoader() noexcept;
  ~JsonVolumeFileLoader() noexcept override;

  const std::string& getName() const noexcept override;

  void process() noexcept override;

  void onMessageFromJS(nlohmann::json const& message) override;

  nlohmann::json getData() const override;
  void           setData(nlohmann::json const& json) override;

 private:
  std::string               mFileName;
  std::shared_ptr<Volume3D> mData;
};

struct JsonVolume {
  glm::uvec3          dimensions{};
  glm::dvec3          origin{};
  glm::dvec3          spacing{};
  std::vector<double> data{};
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_JSON_VOLUME_FILE_LOADER_HPP
