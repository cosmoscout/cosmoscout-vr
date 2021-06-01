////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GRAPHICS_ECLIPSE_SHADOW_MAP_HPP
#define CS_GRAPHICS_ECLIPSE_SHADOW_MAP_HPP

#include <VistaOGLExt/VistaTexture.h>

#include <memory>
#include <string>

namespace cs::graphics {

/// This struct stores information required for each eclipse shadow map. The caster radius
/// includes the height of the atmosphere (if there is any).
struct EclipseShadowMap {
  EclipseShadowMap(
      std::string const& casterAnchor, double casterRadius, std::unique_ptr<VistaTexture>&& texture)
      : mCasterAnchor(casterAnchor)
      , mCasterRadius(casterRadius)
      , mTexture(std::move(texture)) {
  }
  std::string                   mCasterAnchor;
  double                        mCasterRadius;
  std::unique_ptr<VistaTexture> mTexture;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_ECLIPSE_SHADOW_MAP_HPP
