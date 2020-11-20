////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GRAPHICS_COLOR_MAP_HPP
#define CS_GRAPHICS_COLOR_MAP_HPP

#include "cs_graphics_export.hpp"

#include <VistaOGLExt/VistaTexture.h>

#include <filesystem>
#include <memory>
#include <string>

namespace cs::graphics {

/// A color map specified by a json file.
class CS_GRAPHICS_EXPORT ColorMap {
 public:
  explicit ColorMap(std::string const& sJsonString);
  explicit ColorMap(std::filesystem::path const& sJsonPath);

  /// Binds the color map for use in rendering.
  void bind(unsigned unit);

  /// Unbinds the color map after rendering.
  void unbind(unsigned unit);

 private:
  int                           mResolution = 256;
  std::unique_ptr<VistaTexture> mTexture;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_COLOR_MAP_HPP
