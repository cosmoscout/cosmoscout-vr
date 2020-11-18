////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_ATMOSPHERE_PLUGIN_HPP
#define CSP_ATMOSPHERE_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-utils/Property.hpp"

class VistaOpenGLNode;

namespace csp::atmospheres {

class Atmosphere;

/// This plugin adds atmospheres to planets and moons. It uses mie and rayleigh scattering for
/// rendering atmospheric effects. It is configurable via the application config file. See README.md
/// for details.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    struct Atmosphere {
      float                      mAtmosphereHeight{}; ///< Relative to the planets radius.
      float                      mMieHeight{};
      float                      mMieScatteringR{};
      float                      mMieScatteringG{};
      float                      mMieScatteringB{};
      float                      mMieAnisotropy{};
      float                      mRayleighHeight{};
      float                      mRayleighScatteringR{};
      float                      mRayleighScatteringG{};
      float                      mRayleighScatteringB{};
      float                      mRayleighAnisotropy{};
      std::optional<std::string> mCloudTexture; ///< Path to the cloud texture.
      std::optional<float>       mCloudHeight;  ///< Relative to the planets radius.
    };

    std::map<std::string, Atmosphere> mAtmospheres;

    cs::utils::DefaultProperty<bool>  mEnabled{true};
    cs::utils::DefaultProperty<int>   mQuality{7};
    cs::utils::DefaultProperty<float> mWaterLevel{0.F};
    cs::utils::DefaultProperty<bool>  mEnableClouds{true};
    cs::utils::DefaultProperty<bool>  mEnableLightShafts{false};
    cs::utils::DefaultProperty<bool>  mEnableWater{false};
  };

  void init() override;
  void deInit() override;

  void update() override;

 private:
  void onLoad();

  std::shared_ptr<Settings>                          mPluginSettings = std::make_shared<Settings>();
  std::map<std::string, std::shared_ptr<Atmosphere>> mAtmospheres;

  int mEnableShadowsConnection     = -1;
  int mEnableHDRConnection         = -1;
  int mAmbientBrightnessConnection = -1;
  int mOnLoadConnection            = -1;
  int mOnSaveConnection            = -1;
};

} // namespace csp::atmospheres

#endif // CSP_ATMOSPHERE_PLUGIN_HPP
