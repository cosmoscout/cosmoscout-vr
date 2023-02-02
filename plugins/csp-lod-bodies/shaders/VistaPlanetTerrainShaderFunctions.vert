////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

layout(location = 0) in ivec2 VP_iPosition;

float VP_getJR(vec2 posXY)
{
    return VP_f1f2.x - posXY.x - posXY.y;
}

// Returns the texture coordinates inside the tile in [0..1]. The coordinates are clamped to the
// skirt. This means, the coordinates range from [0..1] on top of the tile, the additional skirt
// vertices have the same coordinates as the vertices at the tile boundary.
vec2 VP_getTexCoord(vec2 iPosition)
{
    return clamp((iPosition - vec2(1.0)) / (VP_resolution - 1), vec2(0.0), vec2(1.0));
}

//  Calculates the position (in [0,1]^2) relative to the base patch from
//  integer vertex coordinates @a vtxPos (in [0,256]^2).
vec2 VP_getXY(ivec2 iPosition)
{
    // First convert vtxPos to a relative position ([0,1]^2) within the patch.
    // Then apply VP_offsetScale to obtain relative position within the
    // base patch.
    return (VP_getTexCoord(iPosition) + VP_offsetScale.xy) / VP_offsetScale.z;
}

// Returns the height in meters of the vertex at the given position inside the current tile. The
// outer-most ring of vertices create the skirt around the tile and are therefore moved down. The
// distance which they are moved, depends on the topography of the tile: Flat tiles have a smaller
// skirt than mountainous tiles.
float VP_getVertexHeight(ivec2 iPosition)
{
    vec2 tc = VP_getTexCoord(iPosition);
    float height = texture(VP_texDEM, vec3(tc, VP_dataLayers.x)).x;

    // Move skirt vertices down by half the maximum elevation difference inside the tile.
    if (any(equal(iPosition, ivec2(0.0))) || any(equal(iPosition, ivec2(VP_resolution + 1)))) {
        height -= VP_heightInfo.y * 0.5;
    }

    return height;
}

// Converts point posXY (in [0, 1]^2), which are relative coordinates inside a
// base patch to geodetic coordinates (lat, lng).
// The relevant base patch is indirectly specified through the uniform
// variable VP_f1f2.
vec2 VP_convertXY2lnglat(vec2 posXY)
{
    vec2  result;
    float jr = VP_getJR(posXY);
    float nr = 1.0;

    if(jr < 1.0)
    {
        nr = jr;
        result.y = 1.0 - (nr * nr / 3.0);
    }
    else if(jr > 3.0)
    {
        nr = 4.0 - jr;
        result.y = (nr * nr / 3.0) - 1.0;
    }
    else
    {
        nr = 1.0;
        result.y = (2.0 - jr) * 2.0 / 3.0;
    }

    float tmp = VP_f1f2.y * nr + posXY.x - posXY.y;
    tmp = (tmp <  0.0) ? tmp + 8.0 : tmp;
    tmp = (tmp >= 8.0) ? tmp - 8.0 : tmp;

    result.x = ((nr < 1e-15) ? 0.0 : (VP_PI/4 * tmp) / nr) - VP_PI;
    result.y = VP_PI/2 - acos(result.y);

    return result;
}

vec3 VP_toNormal(vec2 lnglat) {
  return vec3(cos(lnglat.y) * sin(lnglat.x), 
              sin(lnglat.y),
              cos(lnglat.y) * cos(lnglat.x));
}

// Converts point @a lnglat from geodetic (lat,lng) to cartesian
// coordinates (x,y,z) for an ellipsoid with radii @a radii.
vec3 VP_toCartesian(vec2 lnglat, vec3 radii)
{
  vec3 normal  = VP_toNormal(lnglat);
  vec3 normal2 = normal * normal;
  vec3 radii2  = radii * radii;
  return (radii2 * normal) / sqrt(dot(radii2, normal2));
}

vec3 VP_getVertexPositionHEALPix(ivec2 iPosition)
{
    vec2  posXY  = VP_getXY(iPosition);
    vec2  lnglat = VP_convertXY2lnglat(posXY);
    float height = VP_heightScale * VP_getVertexHeight(iPosition);
    vec3  normal = VP_toNormal(lnglat);
    vec3  posXYZ = VP_toCartesian(lnglat, VP_radii);

    posXYZ += height * normal;

    return (VP_matModel * vec4(posXYZ, 1)).xyz;
}

vec3 VP_getVertexPositionInterpolated(ivec2 iPosition)
{
    //   direction   index      alpha
    //  
    //       N         0         1,1           
    //     W   E     1   3    0,1   1,0           
    //       S         2         0,0           
    //  
    vec2 alpha = VP_getTexCoord(iPosition);

    // calculate normal direction by slerping
    vec3 normalSW = mix(VP_normals[2], VP_normals[1], alpha.y);
    vec3 normalNE = mix(VP_normals[3], VP_normals[0], alpha.y);
    vec3 normal   = mix(normalSW, normalNE, alpha.x);

    // calculate height above surface
    // average height is substracted in order to increase the accuracy on high mountains and in deep valleys
    float height = length(VP_matModel[0]) * VP_heightScale 
                   * (VP_getVertexHeight(iPosition) - VP_heightInfo.x);

    // calculate final position
    vec3 result = height * normalize(normal);

    if (alpha.x + alpha.y < 1.0)
    {
        // southern triangle
        result += VP_corners[2] + (VP_corners[3] - VP_corners[2]) * alpha.x
                                + (VP_corners[1] - VP_corners[2]) * alpha.y;
    }
    else
    {
        // northern triangle
        result += VP_corners[0] + (VP_corners[1] - VP_corners[0]) * (1-alpha.x)
                                + (VP_corners[3] - VP_corners[0]) * (1-alpha.y);
    }

    return result;
}

// Given integer vertex coordinates @a vtxPos calculate the model space
// position of the vertex taking into account the elevation data sampled from
// VP_texDEM.
vec3 VP_getVertexPosition(ivec2 iPosition, int mode)
{
    if (mode == 0) {
        return VP_getVertexPositionHEALPix(iPosition);
    } else if (mode == 1) {
        return VP_getVertexPositionInterpolated(iPosition);
    } else {
        vec3 farPos = VP_getVertexPositionHEALPix(iPosition);
        vec3 nearPos = VP_getVertexPositionInterpolated(iPosition);

        vec3 center = (VP_matModel * vec4(0, 0, 0, 1)).xyz;
        float distCenter = length(center);
        float distSurface = length(nearPos);
        float alpha = (distSurface/distCenter - VP_blendEnd) / (VP_blendStart-VP_blendEnd);

        return mix(nearPos, farPos, clamp(alpha, 0, 1));
    }

}

// Given integer vertex coordinates @a vtxPos calculate model space normal
// for the vertex taking into account the elevation data sampled from
// VP_texDEM.
vec3 VP_getVertexNormal(ivec2 iPosition, int mode)
{
    // Make sure to handle bottom skirt vertices the same as top skirt vertices.
    iPosition = clamp(iPosition, ivec2(1), ivec2(VP_resolution));

    // neighbour vertices (p: positive direction, n: negative direction)
    ivec2 pp = ivec2(min(iPosition.x + 1, VP_resolution), iPosition.y);
    ivec2 nn = ivec2(max(iPosition.x - 1, 1), iPosition.y);
    ivec2 np = ivec2(iPosition.x, min(iPosition.y + 1, VP_resolution));
    ivec2 pn = ivec2(iPosition.x, max(iPosition.y - 1, 1));

    // euclidian position of neighbour vertices
    vec3 p_pp = VP_getVertexPosition(pp, mode);
    vec3 p_nn = VP_getVertexPosition(nn, mode);
    vec3 p_np = VP_getVertexPosition(np, mode);
    vec3 p_pn = VP_getVertexPosition(pn, mode);

    // central differences as approximation for surface tangents
    vec3 dx = p_pp - p_nn;
    vec3 dy = p_np - p_pn;

    // cross product of tangents -> normal
    return normalize(cross(dx, dy));
}

// Given integer vertex coordinates @a vtxPos calculate model space normal
// for the vertex taking into account the elevation data sampled from
// VP_texDEM.
vec3 VP_getVertexNormalLow(vec3 centerPos, ivec2 iPosition, int mode)
{
    // Make sure to handle bottom skirt vertices the same as top skirt vertices.
    iPosition = clamp(iPosition, ivec2(1), ivec2(VP_resolution));

    // Sample neighbour vertices. If we are close to a border, we sample in the other direction
    // instead. At iPosition.x == 0 is the bottom skirt vertex.
    ivec2 px = ivec2(iPosition.x == 1 ? 2 : iPosition.x - 1, iPosition.y);
    ivec2 py = ivec2(iPosition.x, iPosition.y == 1 ? 2 : iPosition.y - 1);

    // euclidian position of neighbour vertices
    vec3 p_px = VP_getVertexPosition(px, mode);
    vec3 p_py = VP_getVertexPosition(py, mode);

    // central differences as approximation for surface tangents
    vec3 dx = p_px - centerPos;
    vec3 dy = p_py - centerPos;

    // If we sampled close to either of the sides, we flipped the lookup and hence have to flip the
    // normal as well.
    if (iPosition.x == 1 ^^ iPosition.y == 1) {
        dx = -dx;
    }

    // cross product of tangents -> normal
    return normalize(cross(dx, dy));
}
