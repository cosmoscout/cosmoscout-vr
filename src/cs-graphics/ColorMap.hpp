////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_GRAPHICS_COLOR_MAP_HPP
#define CS_GRAPHICS_COLOR_MAP_HPP

#include "cs_graphics_export.hpp"

#include <VistaOGLExt/VistaTexture.h>
#include <glm/glm.hpp>

#include <filesystem>
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
  explicit ColorMap(std::filesystem::path const& sJsonPath);

  /// Binds the color map for use in rendering.
  void bind(unsigned unit);

  /// Unbinds the color map after rendering.
  void unbind(unsigned unit);

  /// Returns the color map as a vector of RGBA values.
  std::vector<glm::vec4> getRawData();

  /// Returns true if the alpha channel of the color map is not everywhere set to one.
  bool getUsesAlpha() const;

 private:
  int                           mResolution = 256;
  std::unique_ptr<VistaTexture> mTexture;
  std::vector<glm::vec4>        mRawData;
  bool                          mUsesAlpha = false;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_COLOR_MAP_HPP
