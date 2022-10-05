////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#version 330

// uniforms --------------------------------------------------------------------
uniform mat4 VP_matProjection;
uniform vec3 VP_corners[8];

// inputs ----------------------------------------------------------------------
layout(location = 0) in int index;

// -----------------------------------------------------------------------------
void main()
{
    gl_Position = VP_matProjection * vec4(VP_corners[index], 1.0);
}
