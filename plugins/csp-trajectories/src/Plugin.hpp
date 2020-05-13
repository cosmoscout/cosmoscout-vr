////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_TRAJECTORIES_PLUGIN_HPP
#define CSP_TRAJECTORIES_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-utils/Property.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <optional>

namespace csp::trajectories {

class DeepSpaceDot;
class SunFlare;
class Trajectory;

/// This plugin is providing HUD elements that display trajectories and markers for orbiting
/// objects. The configuration of this plugin is done via the provided json config. See README.md
/// for details.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {

    /// The root settings for a single trajectory.
    struct Trajectory {
      /// Settings for a trail behind an object.
      struct Trail {

        /// Without this, some versions of clang claim this type to be non-default-constructible...
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

      /// If available and true a dot will indicate the objects position.
      std::optional<bool> mDrawDot;

      /// If available and true a flare will be drawn around the object.
      std::optional<bool> mDrawFlare;

      /// If available a trail will be drawn behind the object.
      std::optional<Trail> mTrail;
    };

    /// All trajectories with their name as key.
    std::map<std::string, Trajectory> mTrajectories;

    /// Toggles trajectories at runtime.
    cs::utils::DefaultProperty<bool> mEnableTrajectories{true};

    /// Toggles flares at runtime.
    cs::utils::DefaultProperty<bool> mEnableSunFlares{true};

    /// Toggles dots at runtime.
    cs::utils::DefaultProperty<bool> mEnablePlanetMarks{true};
  };

  void init() override;
  void deInit() override;

 private:
  void onLoad();

  std::shared_ptr<Settings>                          mPluginSettings = std::make_shared<Settings>();
  std::map<std::string, std::shared_ptr<Trajectory>> mTrajectories;
  std::map<std::string, std::shared_ptr<DeepSpaceDot>> mDeepSpaceDots;
  std::map<std::string, std::shared_ptr<SunFlare>>     mSunFlares;

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};

} // namespace csp::trajectories

#endif // CSP_TRAJECTORIES_PLUGIN_HPP
