////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_TRAJECTORIES_PLUGIN_HPP
#define CSP_TRAJECTORIES_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-utils/Property.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <optional>

namespace csp::trajectories {

class DeepSpaceDot;
class Trajectory;

/// This plugin is providing HUD elements that display trajectories and markers for orbiting
/// objects. It also draws some flares around objects in HDR and non-HDR mode so that bodies
/// are visible even if they are smaller than a pixel.
/// The configuration of this plugin is done via the provided json config. See README.md
/// for details.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {

    /// The root settings for a single trajectory.
    struct Trajectory {
      /// Settings for a trail behind an object.
      struct Trail {

        /// Without this, some versions of clang claim this type to be non-default-constructible...
        /// Also, Trail() = default; does not work...
        Trail() {
        }

        /// The length of the trail in days.
        double mLength{};

        /// The amount of samples that make up the trail. The higher the better it looks, but the
        /// worse the performance gets.
        int32_t mSamples{};

        /// The name of the anchor this trail is drawn relative to.
        std::string mParent;
      };

      /// Specifies the color of the trail and dot.
      glm::vec3 mColor{};

      /// If true, a dot will indicate the objects position.
      cs::utils::DefaultProperty<bool> mDrawDot{true};

      /// If true, a flare will be drawn around the object in non-HDR mode.
      cs::utils::DefaultProperty<bool> mDrawLDRFlare{false};

      /// If true, a flare will be drawn around the object in HDR mode.
      cs::utils::DefaultProperty<bool> mDrawHDRFlare{true};

      /// Specifies the color of the flare.
      cs::utils::DefaultProperty<glm::vec3> mFlareColor{glm::vec3(1.F, 1.F, 1.F)};

      /// If available a trail will be drawn behind the object.
      std::optional<Trail> mTrail;
    };

    /// All trajectories with their name as key.
    std::map<std::string, Trajectory> mTrajectories;

    /// Toggles trajectories at runtime.
    cs::utils::DefaultProperty<bool> mEnableTrajectories{true};

    /// Toggles non-HDR flares at runtime.
    cs::utils::DefaultProperty<bool> mEnableLDRFlares{true};

    /// Toggles HDR flares at runtime.
    cs::utils::DefaultProperty<bool> mEnableHDRFlares{true};

    /// Toggles dots at runtime.
    cs::utils::DefaultProperty<bool> mEnablePlanetMarks{true};
  };

  void init() override;
  void deInit() override;
  void update() override;

 private:
  void onLoad();
  void onSave();

  std::shared_ptr<Settings>                  mPluginSettings = std::make_shared<Settings>();
  std::vector<std::unique_ptr<Trajectory>>   mTrajectories;
  std::vector<std::unique_ptr<DeepSpaceDot>> mTrajectoryDots;
  std::vector<std::unique_ptr<DeepSpaceDot>> mHDRFlares;
  std::vector<std::unique_ptr<DeepSpaceDot>> mLDRFlares;

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};

} // namespace csp::trajectories

#endif // CSP_TRAJECTORIES_PLUGIN_HPP
