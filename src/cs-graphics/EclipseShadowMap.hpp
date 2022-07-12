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

/// This struct stores information required for each eclipse shadow map. This is an anchor name (as
/// used by the core::Settings::getAnchor* methods) as well as the actual shadow texture.
struct EclipseShadowMap {
  std::string                   mOccluder;
  std::shared_ptr<VistaTexture> mTexture;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_ECLIPSE_SHADOW_MAP_HPP
