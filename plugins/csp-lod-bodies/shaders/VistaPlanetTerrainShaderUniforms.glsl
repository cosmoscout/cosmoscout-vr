////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// constants -------------------------------------------------------------------
const float VP_PI = 3.1415926535897932384626433832795;

// uniforms - global for a planet ----------------------------------------------
uniform mat4  VP_matProjection;
uniform mat4  VP_matModel;
uniform mat4  VP_matView;

uniform float VP_blendEnd = 0.0002;
uniform float VP_blendStart = 0.02;

// planet parameters, radius and scale factor for height values
uniform float VP_heightScale;
uniform vec3 VP_radii;

// texture storing elevation data for all patches
uniform sampler2DArray VP_texDEM;
uniform sampler2DArray VP_texIMG;

// uniforms - current tile -----------------------------------------------------

// The first component contains the average height value of the tile.
// The second component contains the maximum height difference in the tile.
uniform vec2 VP_heightInfo;

// offset (xy) and total number of patches (z) (relative to base patch)
uniform ivec3 VP_offsetScale;

// patch coordinate parameters f1, f2 (indirectly specifies base patch)
uniform ivec2 VP_f1f2;

// Layers of VP_texDEM and VP_texIMG where the current patch's elevation (.x) and image
// data (.y) are stored.
uniform ivec2 VP_dataLayers;

uniform vec3 VP_corners[4];
uniform vec3 VP_normals[4];

// uniforms - shadow stuff -----------------------------------------------------
uniform bool            VP_shadowMapMode;
uniform sampler2DShadow VP_shadowMaps[5];
uniform mat4            VP_shadowProjectionViewMatrices[5];
uniform float           VP_shadowBias = 0.0001;
uniform int             VP_shadowCascades;

// Returns the resolution of the square-shaped image tiles.
int VP_getResolutionIMG() {
    return textureSize(VP_texIMG, 0).x;
}

// Returns the resolution of the square-shaped elevation tiles.
int VP_getResolutionDEM() {
    return textureSize(VP_texDEM, 0).x;
}