////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#version 330

in vec3 position;

// outputs ---------------------------------------------------------------------
layout(location = 0) out vec4 color;

// -----------------------------------------------------------------------------
void main(void)
{
    color = vec4(0.0, 1.0, 0.0, 1.0);
}
