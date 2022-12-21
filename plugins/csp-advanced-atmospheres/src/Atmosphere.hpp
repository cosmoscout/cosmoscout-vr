////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ATMOSPHERES_ATMOSPHERE_HPP
#define CSP_ATMOSPHERES_ATMOSPHERE_HPP

#include "../../../src/cs-scene/CelestialObject.hpp"
#include "AtmosphereRenderer.hpp"
#include "Plugin.hpp"

namespace cs::core {
class SolarSystem;
}

namespace csp::atmospheres {

/// This is a wrapper around a AtmosphereRenderer, adding SPICE based positioning.
class Atmosphere {
 public:
  Atmosphere(std::shared_ptr<Plugin::Settings> pluginSettings,
      std::shared_ptr<cs::core::Settings>      settings,
      std::shared_ptr<cs::core::SolarSystem> solarSystem, std::string objectName);
  ~Atmosphere();

  Atmosphere(Atmosphere const& other) = delete;
  Atmosphere(Atmosphere&& other)      = delete;

  Atmosphere& operator=(Atmosphere const& other) = delete;
  Atmosphere& operator=(Atmosphere&& other) = delete;

  /// Configures the internal renderer according to the given values.
  void configure(Plugin::Settings::Atmosphere const& settings);

  /// Access the internal AtmosphereRender to configure atmosphere parameters.
  AtmosphereRenderer&       getRenderer();
  AtmosphereRenderer const& getRenderer() const;

  /// This is called once a frame by the plugin. It updates the EclipseShadowReceiver.
  void update();

 private:
  std::shared_ptr<Plugin::Settings>                mPluginSettings;
  std::shared_ptr<cs::core::Settings>              mAllSettings;
  std::shared_ptr<cs::core::SolarSystem>           mSolarSystem;
  std::string                                      mObjectName;
  std::shared_ptr<cs::core::EclipseShadowReceiver> mEclipseShadowReceiver;
  AtmosphereRenderer                               mRenderer;
  std::unique_ptr<VistaOpenGLNode>                 mAtmosphereNode;
};

} // namespace csp::atmospheres

#endif // CSP_ATMOSPHERES_ATMOSPHERE_HPP
