////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_TOOLS_GEOMETRIES_HPP
#define CSL_TOOLS_GEOMETRIES_HPP

#include "csl_tools_export.hpp"

namespace csl::tools::geometries {

// Uses glDrawElements with GL_TRIANGLES to draw an ico sphere. It has a radius of 1.0 and is
// centered at the origin.
CSL_TOOLS_EXPORT void drawSphere();

// Uses glDrawElements with GL_TRIANGLES to draw a cube. It has an edge length of 2.0 and is
// centered at the origin.
CSL_TOOLS_EXPORT void drawCube();

} // namespace csl::tools::geometries

#endif // CSL_TOOLS_GEOMETRIES_HPP
