#version 430

////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// ==========================================================================

layout(vertices = 3) out;

uniform int Tesselation_Level = 1;

// inputs
// ==========================================================================
in VS_OUT
{
    vec2  texcoords;
    vec3  normal;
    vec3  position;
    vec3  planetCenter;
    vec2  lngLat;
    float height;
    vec2  vertexPosition;
    vec3  sunDir;
} tc_in[];

// outputs
// ==========================================================================

out TC_OUT
{
    vec2  texcoords;
    vec3  normal;
    vec3  position;
    vec3  planetCenter;
    vec2  lngLat;
    float height;
    vec2  vertexPosition;
    vec3  sunDir;
} tc_out[];

// main
// ==========================================================================

void main()
{
    // Tesselation - Inner and Outer
    // Outer are the three edges
    // Inner is one value only

    gl_TessLevelInner[0] = Tesselation_Level;

    gl_TessLevelOuter[0] = Tesselation_Level;
    gl_TessLevelOuter[1] = Tesselation_Level;
    gl_TessLevelOuter[2] = Tesselation_Level;

    // Pass Through Data
    tc_out[gl_InvocationID].texcoords      = tc_in[gl_InvocationID].texcoords;
    tc_out[gl_InvocationID].normal         = tc_in[gl_InvocationID].normal;
    tc_out[gl_InvocationID].position       = tc_in[gl_InvocationID].position;
    tc_out[gl_InvocationID].planetCenter   = tc_in[gl_InvocationID].planetCenter;
    tc_out[gl_InvocationID].lngLat         = tc_in[gl_InvocationID].lngLat;
    tc_out[gl_InvocationID].height         = tc_in[gl_InvocationID].height;
    tc_out[gl_InvocationID].vertexPosition = tc_in[gl_InvocationID].vertexPosition;
    tc_out[gl_InvocationID].sunDir         = tc_in[gl_InvocationID].sunDir;

    gl_out[gl_InvocationID].gl_Position    = tc_in[gl_InvocationID].gl_Position;
}