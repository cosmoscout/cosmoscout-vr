#version 430

////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// ==========================================================================
// include helper functions/declarations from VistaPlanet
$VP_TERRAIN_SHADER_UNIFORMS
$VP_TERRAIN_SHADER_FUNCTIONS

// uniforms
uniform vec4 uSunDirIlluminance;

// outputs
// ==========================================================================

out VS_OUT
{
    vec2  tileCoords;
    vec3  normal;
    vec3  position;
    vec3  planetCenter;
    vec2  lngLat;
    float height;
    vec2  vertexPosition;
    vec3  sunDir;
} vsOut;

// main
// ==========================================================================

void main(void)
{
    // all in view space
    vsOut.position = VP_getVertexPosition(VP_iPosition, $TERRAIN_PROJECTION_TYPE);
    gl_Position    = VP_matProjection * VP_matView * vec4(vsOut.position, 1);

    if (!VP_shadowMapMode)
    {
        #if $LIGHTING_QUALITY > 2
            vsOut.normal         = VP_getVertexNormal(VP_iPosition, $TERRAIN_PROJECTION_TYPE);
        #elif $LIGHTING_QUALITY > 1
            vsOut.normal         = VP_getVertexNormalLow(vsOut.position, VP_iPosition, $TERRAIN_PROJECTION_TYPE);
        #endif
        vsOut.sunDir         = (VP_matModel * vec4(uSunDirIlluminance.xyz, 0)).xyz;
        vsOut.planetCenter   = (VP_matModel * vec4(0,0,0,1)).xyz;
        vsOut.tileCoords     = VP_getTileCoords(VP_iPosition);
        vsOut.height         = VP_getVertexHeight(VP_iPosition);
        vsOut.lngLat         = VP_convertXY2lnglat(VP_getXY(VP_iPosition));
        vsOut.vertexPosition = VP_iPosition;
    }
}
