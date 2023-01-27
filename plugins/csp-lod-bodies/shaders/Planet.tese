#version 430

////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

// ==========================================================================

layout(triangles) in;

// inputs
// ==========================================================================
in TC_OUT
{
    vec2  texcoords;
    vec3  normal;
    vec3  position;
    vec3  planetCenter;
    vec2  lngLat;
    float height;
    vec2  vertexPosition;
    vec3  sunDir;
} te_in[];

// outputs
// ==========================================================================

out TE_OUT
{
    vec2  texcoords;
    vec3  normal;
    vec3  position;
    vec3  planetCenter;
    vec2  lngLat;
    float height;
    vec2  vertexPosition;
    vec3  sunDir;
} te_out;

// main
// ==========================================================================

void main()
{
    // Interpolation using Barycentric coordinates (triangle topology)
    vec4 p1 = vec4(0.0);

    p1 += gl_TessCoord.x * gl_in[0].gl_Position;
    p1 += gl_TessCoord.y * gl_in[1].gl_Position;
    p1 += gl_TessCoord.z * gl_in[2].gl_Position;

    vec2 tc = vec2(0.0);

    tc += gl_TessCoord.x * te_in[0].texcoords;
    tc += gl_TessCoord.y * te_in[1].texcoords;
    tc += gl_TessCoord.z * te_in[2].texcoords;

    vec3 n = vec3(0.0);

    n += gl_TessCoord.x * te_in[0].normal;
    n += gl_TessCoord.y * te_in[1].normal;
    n += gl_TessCoord.z * te_in[2].normal;

    vec3 p0 = vec3(0.0);

    p0 += gl_TessCoord.x * te_in[0].position;
    p0 += gl_TessCoord.y * te_in[1].position;
    p0 += gl_TessCoord.z * te_in[2].position;

    vec3 pc = vec3(0.0);

    pc += gl_TessCoord.x * te_in[0].planetCenter;
    pc += gl_TessCoord.y * te_in[1].planetCenter;
    pc += gl_TessCoord.z * te_in[2].planetCenter;

    vec2 ll = vec2(0.0);

    ll += gl_TessCoord.x * te_in[0].lngLat;
    ll += gl_TessCoord.y * te_in[1].lngLat;
    ll += gl_TessCoord.z * te_in[2].lngLat;

    float h = float(0.0);

    h += gl_TessCoord.x * te_in[0].height;
    h += gl_TessCoord.y * te_in[1].height;
    h += gl_TessCoord.z * te_in[2].height;

    vec2 vp = vec2(0.0);

    vp += gl_TessCoord.x * te_in[0].vertexPosition;
    vp += gl_TessCoord.y * te_in[1].vertexPosition;
    vp += gl_TessCoord.z * te_in[2].vertexPosition;

    vec3 sd = vec3(0.0);

    sd += gl_TessCoord.x * te_in[0].sunDir;
    sd += gl_TessCoord.y * te_in[1].sunDir;
    sd += gl_TessCoord.z * te_in[2].sunDir;

    te_out.texcoords      = tc;
    te_out.normal         = normalize(n);
    te_out.position       = p0;
    te_out.planetCenter   = pc;
    te_out.lngLat         = ll;
    te_out.height         = h;
    te_out.vertexPosition = vp;
    te_out.sunDir         = sd;

    // te_out.texcoords      = te_in[0].texcoords;
    // te_out.normal         = te_in[0].normal;
    // te_out.position       = te_in[0].position;
    // te_out.planetCenter   = te_in[0].planetCenter;
    // te_out.lngLat         = te_in[0].lngLat;
    // te_out.height         = te_in[0].height;
    // te_out.vertexPosition = te_in[0].vertexPosition;
    // te_out.sunDir         = te_in[0].sunDir;

    // final position
    gl_Position = p1;
}