////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_SIMPLE_BODIES_PLUGIN_HPP
#define CSP_SIMPLE_BODIES_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-utils/DefaultProperty.hpp"

#include <glm/glm.hpp>

#include <map>
#include <optional>
#include <string>

namespace csp::simplebodies {

class SimpleBody;

/// This plugin provides the rendering of planets as spheres with a texture. Despite its name it
/// can also render moons :P. It can be configured via the applications config file. See README.md
/// for details.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    struct SimpleBody {
      std::string                           mTexture;
      cs::utils::DefaultProperty<bool>      mPrimeMeridianInCenter{true};
      cs::utils::DefaultProperty<glm::vec3> mAlbedo{glm::vec3(1.F, 1.F, 1.F)};

      struct Ring {
        /// The path to the texture. The texture should represent a cross section of the ring.
        std::string mTexture;

        /// The distance from the planet's center to where the ring starts in meters.
        double mInnerRadius;

        /// The distance from the planet's center to where the ring ends in meters.
        double mOuterRadius;
      };

      std::optional<Ring> mRing;
    };

    std::map<std::string, SimpleBody> mSimpleBodies;
  };

  void init() override;
  void deInit() override;
  void update() override;

 private:
  void onLoad();
  void onSave();
  void unregisterBody(std::string const& name);

  Settings                                           mPluginSettings;
  std::map<std::string, std::shared_ptr<SimpleBody>> mSimpleBodies;

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};

} // namespace csp::simplebodies

#endif // CSP_SIMPLE_BODIES_PLUGIN_HPP
