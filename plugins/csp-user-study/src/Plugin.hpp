////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_USER_STUDY_PLUGIN_HPP
#define CSP_USER_STUDY_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-utils/Property.hpp"
#include <vector>

namespace cs::scene {
  class CelestialAnchorNode;
} // namespace cs::scene

namespace csp::userstudy {
class UserStudy;

/// This plugin creates configurable navigation scenarios for a user study.
/// It uses scaleable web views to mark checkpoints with different tasks.
/// The plugin is configurable via the application config file. See README.md for details.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    /// Enum for Stage Types
    enum class StageType { eNone, eCheckpoint, eRequestFMS, eSwitchScenario };

    /// Toggle, whether scenario stages should be displayed
    cs::utils::DefaultProperty<bool> mEnabled{false};

    /// Toggle, whether all scenario stages should be displayed all the time
    cs::utils::DefaultProperty<bool> mDebug{false};

    /// The settings for a scenario
    struct Scenario {

      /// The name of the scenario
      cs::utils::DefaultProperty<std::string> mName{"None"};

      /// The path to the scenario config
      cs::utils::DefaultProperty<std::string> mPath{"None"};

      /// Operator to compare two Scenarios
      bool operator==(Scenario const& other) const;
    };
    
    /// List of configs containing related scenarios.
    std::vector<Scenario> mOtherScenarios;

    /// The settings for a stage of the scenario
    struct Stage {
      
      /// The type of the stage
      cs::utils::DefaultProperty<StageType> mType{StageType::eNone};
      
      /// The related bookmark for the position & orientation
      cs::utils::DefaultProperty<std::string> mBookmark{"None"};
      
      /// The scaling factor for the stage mark
      cs::utils::DefaultProperty<float> mScaling{1};

      /// Operator to compare two Stages
      bool operator==(Stage const& other) const;
    };

    /// List of stages making up the scenario
    std::vector<Stage> mStages;

    /// Operator to compare Settings
    bool operator!=(Settings const& other) const;
    /// Operator to compare Settings
    bool operator==(Settings const& other) const;
  };

  void init() override;
  void deInit() override;
  void update() override;

 private:
  void onLoad();
  void unload(Plugin::Settings pluginSettings);

  std::shared_ptr<Settings> mPluginSettings = std::make_shared<Settings>();

  struct Stage {
    std::shared_ptr<cs::scene::CelestialAnchorNode> mAnchor;
    float                                           mScale = 1.0;
  };

  std::list<Stage> mStages = {};

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};
} // namespace csp::userstudy

#endif // CSP_USER_STUDY_PLUGIN_HPP
