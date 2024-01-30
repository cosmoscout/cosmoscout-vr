////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ATMOSPHERES_PLUGIN_HPP
#define CSP_ATMOSPHERES_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-utils/Property.hpp"

class VistaOpenGLNode;

namespace csp::atmospheres {

class Atmosphere;

/// This plugin adds atmospheres to planets and moons. It supports multiple atmospheric models.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    struct Atmosphere {

      /// For now, two different atmospheric models are supported:
      /// - eCosmoScoutVR: A simple fragment-shader raytracer which supports single-scattering and
      ///   can be configured to match various atmospheres, such as Earth's or the one of Mars.
      /// - eBruneton: This is based on the paper "Precomputed Atmospheric Scattering" by Eric
      ///   Bruneton. We generalized the model to accept arbitrary wavelength-dependent phase
      ///   functions and extinction coefficients stored in CSV files. This makes the model more
      ///   versatile and also allows simulation of the Martian atmosphere. The model simulates
      ///   multi-scattering and provides in general a better performance than the CosmoScoutVR
      ///   model. However, under specific circumstances it may exhibit more artifacts due to
      ///   limited floating point precision in the precomputed textures.
      enum class Model { eCosmoScoutVR, eBruneton };

      /// This defines which model should be used by the atmosphere.
      cs::utils::DefaultProperty<Model> mModel{Model::eCosmoScoutVR};

      /// This contains model-specific parameters. The format is defined by the respective model.
      nlohmann::json mModelSettings;

      /// These parameters are model-agnostic.
      double                             mTopAltitude;         ///< In meters.
      cs::utils::DefaultProperty<double> mBottomAltitude{0.0}; ///< In meters.
      cs::utils::DefaultProperty<bool>   mEnableWater{false};
      cs::utils::DefaultProperty<bool>   mEnableWaves{true};
      cs::utils::DefaultProperty<float>  mWaterLevel{0.F}; ///< In meters.
      cs::utils::DefaultProperty<bool>   mEnableClouds{true};
      std::optional<std::string>         mCloudTexture;          ///< Path to the cloud texture.
      cs::utils::DefaultProperty<float>  mCloudAltitude{3000.F}; ///< In meters.

      /// If this is set to true, the plugin will save a fish-eye view of the sky to a file one
      /// the preprocessing is done.
      cs::utils::DefaultProperty<bool> mRenderSkydome{false};
    };

    std::unordered_map<std::string, Atmosphere> mAtmospheres;

    cs::utils::DefaultProperty<bool> mEnable{true};
  };

  /// The plugin uses the standard plugin life cycle. On init, the settings are loaded and the
  /// atmospheres are created. On update, the atmospheres are updated. Finally, on deInit, the
  /// current settings are saved and the atmospheres are destroyed.
  void init() override;
  void deInit() override;
  void update() override;

 private:
  void onLoad();
  void onSave();

  std::shared_ptr<Settings> mPluginSettings = std::make_shared<Settings>();
  std::unordered_map<std::string, std::shared_ptr<Atmosphere>> mAtmospheres;
  std::string                                                  mActiveAtmosphere;

  int mActiveObjectConnection = -1;
  int mOnLoadConnection       = -1;
  int mOnSaveConnection       = -1;
};

} // namespace csp::atmospheres

#endif // CSP_ATMOSPHERES_PLUGIN_HPP
