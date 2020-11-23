////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GRAPHICS_COLOR_MAP_HPP
#define CS_GRAPHICS_COLOR_MAP_HPP

#include "cs_graphics_export.hpp"

#include <VistaOGLExt/VistaTexture.h>
#include <boost/filesystem.hpp>
#include <glm/glm.hpp>

#include <memory>
#include <string>
#include <vector>

namespace cs::graphics {

/// A color map specified by a json file.
class CS_GRAPHICS_EXPORT ColorMap {
 public:
  /// Creates a ColorMap from a json string.
  explicit ColorMap(std::string const& sJsonString);
  /// Creates a ColorMap from the json file at sJsonPath.
  explicit ColorMap(boost::filesystem::path const& sJsonPath);

  /// Binds the color map for use in rendering.
  void bind(unsigned unit);

  /// Unbinds the color map after rendering.
  void unbind(unsigned unit);

  /// Returns the color map as a vector of RGBA values.
  std::vector<glm::vec4> getRawData();

 private:
  int                           mResolution = 256;
  std::unique_ptr<VistaTexture> mTexture;
  std::vector<glm::vec4>        mRawData;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_COLOR_MAP_HPP
