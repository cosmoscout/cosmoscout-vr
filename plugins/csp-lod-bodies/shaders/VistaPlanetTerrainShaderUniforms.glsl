////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

// constants -------------------------------------------------------------------
const float VP_PI = 3.1415926535897932384626433832795;

// texture size
const int   VP_TEXTURESIZE = 257;

// largest integral vertex position
const int   VP_MAXVERTEX      = 256;
const float VP_VERTEXDISTANCE = 1.0 / VP_MAXVERTEX;

// uniforms - global for a planet ----------------------------------------------
uniform mat4  VP_matProjection;
uniform mat4  VP_matModelView;
uniform float VP_farClip;

uniform float VP_blendEnd = 0.0002;
uniform float VP_blendStart = 0.02;

// planet parameters, radius and scale factor for height values
uniform float VP_heightScale;
uniform vec2 VP_radius;

// texture storing elevation data for all patches
uniform sampler2DArray VP_texDEM;
uniform sampler2DArray VP_texIMG;

// uniforms - current tile -----------------------------------------------------

uniform float VP_demAverageHeight;

// offset (xy) and total number of patches (z) (relative to base patch)
uniform ivec3 VP_tileOffsetScale;

// offset (xy) and divisor (z) for DEM tile tex coords
uniform ivec3 VP_demOffsetScale;

// offset (xy) and divisor (z) for IMG tile tex coords
uniform ivec3 VP_imgOffsetScale;

// difference in resolution to neighbour tile (x: NE, y: NW, z: SW, w: SE)
uniform ivec4 VP_edgeDelta;

// layer of VP_texDEM the neighbour tile is stored in (x: NE, y: NW, z: SW,
// w: SE) - only entries VP_edgeLayerDEM.I are valid where VP_edgeDelta.I != 0
uniform ivec4 VP_edgeLayerDEM;

// offset to apply to coordinates on neighbour tiles (x: NE, y: NW, z: SW,
// w: SE)
uniform ivec4 VP_edgeOffset;

// patch coordinate parameters f1, f2 (indirectly specifies base patch)
uniform ivec2 VP_f1f2;

// layer of VP_texDEM the current patch's elevation data is stored in
uniform int VP_layerDEM;

// layer of VP_texIMG the current patch's image data is stored in
uniform int VP_layerIMG;

uniform vec3 VP_corners[4];
uniform vec3 VP_normals[4];

// uniforms - shadow stuff -----------------------------------------------------
uniform bool            VP_shadowMapMode;
uniform sampler2DShadow VP_shadowMaps[5];
uniform mat4            VP_shadowProjectionViewMatrices[5];
uniform float           VP_shadowBias = 0.0001;
uniform int             VP_shadowCascades;
