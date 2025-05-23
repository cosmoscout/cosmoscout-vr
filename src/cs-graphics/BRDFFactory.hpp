////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_GRAPHICS_BRDF_FACTORY_HPP
#define CS_GRAPHICS_BRDF_FACTORY_HPP

#include "cs_graphics_export.hpp"

#include <string>
#include <unordered_map>

namespace cs::graphics {

/// A struct that represents a BRDF, given its source code and material properties.
struct BRDF {
  std::string source; ///< The source code of the BRDF in GLSL-like form.
  std::unordered_map<std::string, float>
      properties; ///< The material properties as key-variables and values.

  BRDF& operator=(BRDF const& other) {
    source     = other.source;
    properties = other.properties;
    return *this;
  }

  bool operator==(BRDF const& other) const {
    return source == other.source && properties == other.properties;
  }
  bool operator!=(BRDF const& other) const {
    return !((*this) == other);
  }
};

class CS_GRAPHICS_EXPORT BRDFFactory {
 private:
  BRDF const brdf;

 public:
  // BRDFFactory(BRDF const& brdf)
  //     : brdf(brdf) {}
  std::string getBRDFSnipped() const;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_BRDF_FACTORY_HPP
