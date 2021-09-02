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

namespace cs::scene {
class CelestialAnchorNode;
} // namespace cs::scene

class VistaOpenGLNode;
class VistaTransformNode;

namespace cs::gui {
class WorldSpaceGuiArea;
class GuiItem;
} // namespace cs::gui

namespace csp::userstudy {

/// This plugin creates configurable navigation scenarios for a user study.
/// It uses scaleable web views to mark checkpoints with different tasks.
/// The plugin is configurable via the application config file. See README.md for details.
class Plugin : public cs::core::PluginBase {
 public:
  enum class StageType { eCheckpoint, eRequestFMS, eRequestCOG, eSwitchScenario };

  struct Settings {

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
    struct StageSetting {

      /// The type of the stage
      cs::utils::DefaultProperty<StageType> mType{StageType::eCheckpoint};

      /// The related bookmark for the position & orientation
      cs::utils::DefaultProperty<std::string> mBookmarkName{"None"};

      /// The scaling factor for the stage mark
      cs::utils::DefaultProperty<float> mScaling{1};

      /// Operator to compare two Stages
      bool operator==(StageSetting const& other) const;
      /// Operator to compare two Stages
      bool operator!=(StageSetting const& other) const;
    };

    /// List of stages making up the scenario
    std::vector<StageSetting> mStageSettings;

    /// The checkpoint recording interval in seconds.
    cs::utils::DefaultProperty<uint32_t> pRecordingInterval{5};

    /// Operator to compare Settings
    bool operator!=(Settings const& other) const;
    /// Operator to compare Settings
    bool operator==(Settings const& other) const;
  };

  void init() override;
  void deInit() override;
  void update() override;

 private:
  void                                        onLoad();
  void                                        unload();
  void                                        setupStage(std::size_t stageIdx);
  std::optional<cs::core::Settings::Bookmark> getBookmarkByName(std::string name);
  void                                        updateStages();
  void                                        advanceStage();

  std::shared_ptr<Settings> mPluginSettings = std::make_shared<Settings>();

  struct Stage {
    std::shared_ptr<cs::scene::CelestialAnchorNode> mAnchor;
    std::unique_ptr<cs::gui::WorldSpaceGuiArea>     mGuiArea;
    std::unique_ptr<VistaTransformNode>             mTransform;
    std::unique_ptr<VistaOpenGLNode>                mGuiNode;
    std::unique_ptr<cs::gui::GuiItem>               mGuiItem;
  };

  std::array<Stage, 3>                  mStages;
  std::size_t                           mStageIdx             = 0;
  bool                                  mEnableRecording      = false;
  bool                                  mEnableCOGMeasurement = false;
  std::chrono::steady_clock::time_point mLastRecordTime;

  cs::utils::Property<uint32_t> mCurrentFMS = 0;

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};
} // namespace csp::userstudy

#endif // CSP_USER_STUDY_PLUGIN_HPP
