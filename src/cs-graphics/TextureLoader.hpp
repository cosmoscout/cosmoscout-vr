////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GRAPHICS_TEXTURE_LOADER_HPP
#define CS_GRAPHICS_TEXTURE_LOADER_HPP

#include "cs_graphics_export.hpp"

#include <VistaOGLExt/VistaTexture.h>
#include <memory>
#include <string>

namespace cs::graphics {
/// For loading VistaTextures.
class CS_GRAPHICS_EXPORT TextureLoader {
 public:
  /// Loads a VistaTexture from the given file.
  static std::unique_ptr<VistaTexture> loadFromFile(std::string const& sFileName);
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_TEXTURE_LOADER_HPP
