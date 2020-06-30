////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_MEASUREMENTTOOLS_PLUGIN_HPP
#define CSP_MEASUREMENTTOOLS_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-utils/DefaultProperty.hpp"

#include <list>
#include <string>

namespace cs::core::tools {
class Tool;
} // namespace cs::core::tools

namespace csp::measurementtools {

class DipStrikeTool;
class EllipseTool;
class FlagTool;
class PathTool;
class PolygonTool;

/// This plugin enables the user to measure different things on the surface of planets and moons.
/// See README.md for details.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {

    std::vector<std::shared_ptr<DipStrikeTool>> mDipStrikes;
    std::vector<std::shared_ptr<EllipseTool>>   mEllipses;
    std::vector<std::shared_ptr<FlagTool>>      mFlags;
    std::vector<std::shared_ptr<PathTool>>      mPaths;
    std::vector<std::shared_ptr<PolygonTool>>   mPolygons;

    cs::utils::DefaultProperty<float>   mPolygonHeightDiff{1.002F};
    cs::utils::DefaultProperty<int32_t> mPolygonMaxAttempt{5};
    cs::utils::DefaultProperty<int32_t> mPolygonMaxPoints{1000};
    cs::utils::DefaultProperty<int32_t> mPolygonSleekness{15};
    cs::utils::DefaultProperty<int32_t> mEllipseSamples{100};
    cs::utils::DefaultProperty<int32_t> mPathSamples{256};
  };

  void init() override;
  void deInit() override;
  void update() override;

 private:
  void onLoad();

  Settings    mPluginSettings{};
  std::string mNextTool = "none";

  int mOnClickConnection       = -1;
  int mOnDoubleClickConnection = -1;
  int mOnLoadConnection        = -1;
  int mOnSaveConnection        = -1;
};

} // namespace csp::measurementtools

#endif // CSP_MEASUREMENTTOOLS_PLUGIN_HPP
