////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_STARS_PLUGIN_HPP
#define CSP_STARS_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-utils/DefaultProperty.hpp"
#include "Stars.hpp"

#include <glm/glm.hpp>
#include <optional>

class VistaOpenGLNode;
class VistaTransformNode;

namespace csp::stars {

/// The starts plugin displays the night sky from star catalogues.
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
    cs::utils::DefaultProperty<std::string>     mCelestialGridTexture{""};
    cs::utils::DefaultProperty<std::string>     mStarFiguresTexture{""};
    cs::utils::DefaultProperty<glm::vec4>       mCelestialGridColor{glm::vec4(0.5F)};
    cs::utils::DefaultProperty<glm::vec4>       mStarFiguresColor{glm::vec4(0.5F)};
    std::string                                 mStarTexture;
    std::optional<std::string>                  mCacheFile;
    std::optional<std::string>                  mHipparcosCatalog;
    std::optional<std::string>                  mTychoCatalog;
    std::optional<std::string>                  mTycho2Catalog;
    std::optional<std::string>                  mGaiaCatalog;
    cs::utils::DefaultProperty<bool>            mEnabled{true};
    cs::utils::DefaultProperty<bool>            mEnableCelestialGrid{false};
    cs::utils::DefaultProperty<bool>            mEnableStarFigures{false};
    cs::utils::DefaultProperty<float>           mLuminanceMultiplicator{0.F};
    cs::utils::DefaultProperty<Stars::DrawMode> mDrawMode{Stars::DrawMode::eGlareDisc};
    cs::utils::DefaultProperty<float>           mSize{0.05F};
    cs::utils::DefaultProperty<glm::vec2>       mMagnitudeRange{glm::vec2(-5.F, 13.F)};
  };

  void init() override;
  void deInit() override;

  void update() override;

 private:
  void onLoad();
  void onSave();

  Settings                            mPluginSettings;
  std::unique_ptr<Stars>              mStars;
  std::unique_ptr<VistaTransformNode> mStarsTransform;
  std::unique_ptr<VistaOpenGLNode>    mStarsNode;

  int mEnableHDRConnection = -1;
  int mOnLoadConnection    = -1;
  int mOnSaveConnection    = -1;
};

} // namespace csp::stars

#endif // CSP_STARS_PLUGIN_HPP
