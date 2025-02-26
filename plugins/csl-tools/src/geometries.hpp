////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_TOOLS_GEOMETRIES_HPP
#define CSL_TOOLS_GEOMETRIES_HPP

#include "csl_tools_export.hpp"

namespace csl::tools::geometries {

// Uses glDrawElements with GL_TRIANGLES to draw an ico sphere. It will have a radius of 1.0 and
// be centered at the origin.
CSL_TOOLS_EXPORT void drawSphere();

} // namespace csl::tools::geometries

#endif // CSL_TOOLS_GEOMETRIES_HPP
