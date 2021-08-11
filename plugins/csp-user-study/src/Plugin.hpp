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
#include <memory>
#include <vector>

namespace csp::userstudy {
class UserStudy;

/// This plugin creates configurable navigation scenarios for a user study.
/// It uses scaleable web views to mark checkpoints with different tasks.
/// The plugin is configurable via the application config file. See README.md for details.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    /// Toggle, whether scenario stages should be displayed
    cs::utils::DefaultProperty<bool> mEnabled{false};

    /// The settings for a scenario
    struct Scenario {

      /// The name of the scenario
      cs::utils::DefaultProperty<std::string> mName{"None"};

      /// The path to the scenario config
      cs::utils::DefaultProperty<std::string> mPath{"None"};
    };
    
    /// List of configs containing related scenarios.
    std::vector<Scenario> mOtherScenarios;

    /// The settings for a stage of the scenario
    struct Stage {
      
      // The type of the stage
      cs::utils::DefaultProperty<std::string> mType{"None"};
      
      // The related bookmark if type is "Checkpoint"
      cs::utils::DefaultProperty<std::string> mBookmark{"None"};
      
      // The scaling factor for the stage mark
      cs::utils::DefaultProperty<float> mScaling{1};
    };

    /// List of stages making up the scenario
    std::vector<Stage> mStages;
  };

  void init() override;
  void deInit() override;
  void update() override;

 private:
  void onLoad();

  std::shared_ptr<Settings>                 mPluginSettings = std::make_shared<Settings>();
  //std::vector<std::unique_ptr<UserStudy>>   mUserStudy;

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};
} // namespace csp::userstudy

#endif // CSP_USER_STUDY_PLUGIN_HPP
