////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#version 330

// uniforms --------------------------------------------------------------------
uniform mat4 VP_matProjection;
uniform vec3 VP_corners[8];

out vec3 position;

// inputs ----------------------------------------------------------------------
layout(location = 0) in int index;

// -----------------------------------------------------------------------------
void main()
{
    position = VP_corners[index];
    gl_Position = VP_matProjection * vec4(VP_corners[index], 1.0);
}
