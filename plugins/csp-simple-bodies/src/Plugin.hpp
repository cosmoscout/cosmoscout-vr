////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_SIMPLE_BODIES_PLUGIN_HPP
#define CSP_SIMPLE_BODIES_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-utils/DefaultProperty.hpp"

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
    /// A struct that represents a BRDF, given its source code and material properties.
    struct BRDF {
      std::string source; ///< The source code of the BRDF in GLSL-like form.
      std::unordered_map<std::string, float>
           properties; ///< The material properties as key-variables and values.
      bool operator==(BRDF const& other) const {
        return source == other.source && properties == other.properties;
      }
      bool operator!=(BRDF const& other) const {
        return !((*this) == other);
      }
    };

    struct SimpleBody {
      std::string                      mTexture;
      cs::utils::DefaultProperty<bool> mPrimeMeridianInCenter{true};

      struct Ring {
        /// The path to the texture. The texture should represent a cross section of the ring.
        std::string mTexture;

        /// The distance from the planet's center to where the ring starts in meters.
        double mInnerRadius;

        /// The distance from the planet's center to where the ring ends in meters.
        double mOuterRadius;
      };

      cs::utils::DefaultProperty<BRDF> mBrdfHdr{
          ///< The BRDF used in HDR mode, with higher precedence than lighting mode.
          BRDF{"../share/resources/shaders/brdfs/lambert.glsl", {{"$rho", 1.0f}}}};
      cs::utils::DefaultProperty<BRDF>  mBrdfNonHdr{///< The BRDF used in lighting mode.
          BRDF{"../share/resources/shaders/brdfs/lambert_scaled.glsl", {{"$rho", 1.0f}}}};
      cs::utils::DefaultProperty<float> mAvgLinearImgIntensity{
          1.0f}; ///< The average intensity of the linear (!) image.

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
