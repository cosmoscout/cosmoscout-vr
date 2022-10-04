////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

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
  /// Loads a VistaTexture from the given file. This support *.tga, *.tif, *.hdr as well as all
  /// image formats supported by stb_image (including *.bmp, *.jpeg and *.png).
  static std::unique_ptr<VistaTexture> loadFromFile(std::string const& sFileName);
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_TEXTURE_LOADER_HPP
