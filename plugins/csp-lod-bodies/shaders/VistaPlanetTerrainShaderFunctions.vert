////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

// -----------------------------------------------------------------------------
// Helper functions for VistaPlanet terrain shaders.
// The two most commonly used functions are 
//      vec4 VP_getVertexPosition(in ivec2 vtxPos)
//      vec3 VP_getVertexNormal(in ivec2 vtxPos)
// near the bottom of the file. They respectively compute the vertex position
// and normal from the elevation data.
// -----------------------------------------------------------------------------

// inputs ----------------------------------------------------------------------

layout(location = 0) in ivec2 VP_iPosition;

float VP_getJR(vec2 posXY)
{
    return VP_f1f2.x - posXY.x - posXY.y;
}

//  Calculates the position (in [0,1]^2) relative to the base patch from
//  integer vertex coordinates @a vtxPos (in [0,256]^2).
vec2 VP_getXY(ivec2 iPosition)
{
    // First convert vtxPos to a relative position ([0,1]^2) within the patch.
    // Then apply VP_tileOffsetScale to obtain relative position within the
    // base patch.
    return ((iPosition + VP_demOffsetScale.xy) * VP_VERTEXDISTANCE + VP_tileOffsetScale.xy) / 
            VP_tileOffsetScale.z;
}

// Find DEM texture coordinates given integer vertex coordinates
// @a vtxPos (in [0,256]^2).
vec2 VP_getTexCoordDEM(vec2 iPosition)
{
    return iPosition / VP_MAXVERTEX;
}

// Find IMG texture coordinates given integer vertex coordinates
// @a vtxPos (in [0,256]^2).
vec2 VP_getTexCoordIMG(vec2 iPosition)
{
    return (iPosition + VP_imgOffsetScale.xy) / VP_imgOffsetScale.z;
}

float VP_getVertexHeight(ivec2 iPosition)
{
    // For edges where the resolution changes, data is taken from the lower
    // resolution neighbour to avoid creating gaps. When the resolution is the
    // same, data is taken from the western neighbour.
    // VP_edgeDelta contains information about the neighbour's resolution
    // (along edge NE, NW, SW, SE in that order). If it is < 0 the neighbour's
    // resolution is lower than this tile's and data from the neighbour should
    // be used.
    // In that case the position along the edge needs to be adjusted (since
    // the neighbour has lower resolution) by applying a scaling factor
    // that depends on the difference in levels (from VP_edgeDelta) and
    // an offset (VP_edgeOffset).
    // If it is == 0, the resolution is the same
    ivec2 basePos = iPosition + VP_demOffsetScale.xy;

    // SW edge - sample neightbour patch if same or lower resolution
    if(basePos.x == 0 && VP_edgeDelta.z <= 0 || basePos.x < 0)
    {
        float scale = 1.0 / (1 << max(0, -VP_edgeDelta.z));

        // determine where on neighbour tile to sample (NE)
        vec2  pos = vec2(basePos.x + VP_MAXVERTEX, scale * basePos.y + VP_edgeOffset.z);
        vec2  tc  = VP_getTexCoordDEM(pos);

        // if we are at an edge of a souther basepatch we need to sample SE
        if (VP_getJR(VP_getXY(iPosition)) >= 3.0 &&          // southern quarter
           VP_tileOffsetScale.x == 0)                        // first tile in x
        {
            tc = vec2(tc.y, 1.0 - tc.x);
        }

        return texture(VP_texDEM, vec3(tc, VP_edgeLayerDEM.z)).x;
    }
    
    // NW edge - sample neightbour patch if same or lower resolution
    if(basePos.y == VP_MAXVERTEX && VP_edgeDelta.y <= 0 || basePos.y > VP_MAXVERTEX)
    {
        float scale = 1.0 / (1 << max(0, -VP_edgeDelta.y));

        // determine where on neighbour tile to sample (SE)
        vec2  pos = vec2(scale * basePos.x + VP_edgeOffset.y, basePos.y - VP_MAXVERTEX);
        vec2  tc  = VP_getTexCoordDEM(pos);

        // if we are at an edge of a northern basepatch we need to sample NE
        if (VP_getJR(VP_getXY(iPosition)) <= 1.0 &&          // northern quarter
           VP_tileOffsetScale.y == VP_tileOffsetScale.z - 1) // last tile in y
        {
            tc = vec2(1.0 - tc.y, tc.x);
        }

        return texture(VP_texDEM, vec3(tc, VP_edgeLayerDEM.y)).x;
    }
    
    // NE edge - sample neightbour patch if lower resolution or if same
    // resolution and position beyond VP_MAXVERTEX is requested
    if((basePos.x == VP_MAXVERTEX && VP_edgeDelta.x < 0) || basePos.x > VP_MAXVERTEX)
    {
        float scale = 1.0 / (1 << max(0, -VP_edgeDelta.x));

        // determine where on neighbour tile to sample (SW)
        vec2  pos = vec2(basePos.x - VP_MAXVERTEX, scale * basePos.y + VP_edgeOffset.x);
        vec2  tc  = VP_getTexCoordDEM(pos);

        // if we are at an edge of a northern basepatch we need to sample NW
        if (VP_getJR(VP_getXY(iPosition)) <= 1.0 &&          // northern quarter
           VP_tileOffsetScale.x == VP_tileOffsetScale.z - 1) // last tile in x
        {
            tc = vec2(tc.y, 1.0 - tc.x);
        }

        return texture(VP_texDEM, vec3(tc, VP_edgeLayerDEM.x)).x;
    }
    
    // SE edge - sample neightbour patch if lower resolution or if same
    // resolution and position beyond 0 is requested
    if((basePos.y == 0 && VP_edgeDelta.w < 0) || basePos.y < 0)
    {
        float scale = 1.0 / (1 << max(0, -VP_edgeDelta.w));

        // determine where on neighbour tile to sample (NW)
        vec2  pos = vec2(scale * basePos.x + VP_edgeOffset.w, basePos.y + VP_MAXVERTEX);
        vec2  tc  = VP_getTexCoordDEM(pos);

        // if we are at an edge of a souther basepatch we need to sample SW
        if (VP_getJR(VP_getXY(iPosition)) >= 3.0 &&          // southern quarter
           VP_tileOffsetScale.y == 0)                        // first tile in y
        {
            tc = vec2(1.0 - tc.y, tc.x);
        }

        return texture(VP_texDEM, vec3(tc, VP_edgeLayerDEM.w)).x;
    }

    // multiple cases here:
    //  non-edge vertex
    //  edge vertex in western direction and neighbours have higher resolution
    //  edge vertex in eastern direction and neighbours have same or higher resolution
    vec2 tc = VP_getTexCoordDEM(basePos);
    return texture(VP_texDEM, vec3(tc, VP_layerDEM)).x;
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

float VP_toGeocentricLat(float geodeticLat, vec2 radius)
{
    float f = (radius.x - radius.y) / radius.x;
    return atan(pow(1.0 - f, 2.0) * tan(geodeticLat));
}

// Converts point @a lnglat from geodetic (lat,lng) to cartesian
// coordinates (x,y,z) for an ellipsoid with radii @a radius.
vec3 VP_toCartesian(vec2 lnglat, vec2 radius)
{
    lnglat.y = VP_toGeocentricLat(lnglat.y, radius);
    
    vec2  c   = cos(lnglat);
    vec2  s   = sin(lnglat);

    // point on ellipsoid surface
    return vec3(c.y * s.x * radius.x,
                  s.y * radius.y,
                  c.y * c.x * radius.x);
}

vec3 VP_toNormal(vec2 lnglat, vec2 radius)
{
    vec3 cart = VP_toCartesian(lnglat, radius);
    return normalize(cart / vec3(radius.x * radius.x, radius.y * radius.y, radius.x * radius.x));
}

vec3 VP_getVertexPositionHEALPix(ivec2 iPosition)
{
    vec2  posXY  = VP_getXY(iPosition);
    vec2  lnglat = VP_convertXY2lnglat(posXY);
    float height = VP_heightScale * VP_getVertexHeight(iPosition);
    vec3 normal  = VP_toNormal(lnglat, VP_radius);
    vec3  posXYZ = VP_toCartesian(lnglat, VP_radius);

    posXYZ += height * normal;

    return (VP_matModelView * vec4(posXYZ, 1)).xyz;
}

vec3 VP_getVertexPositionInterpolated(ivec2 iPosition)
{
    //   direction   index      alpha
    //  
    //       N         0         1,1           
    //     W   E     1   3    0,1   1,0           
    //       S         2         0,0           
    //  
    vec2 alpha = vec2(iPosition) / VP_demOffsetScale.z;

    // calculate normal direction by slerping
    vec3 normalSW = mix(VP_normals[2], VP_normals[1], alpha.y);
    vec3 normalNE = mix(VP_normals[3], VP_normals[0], alpha.y);
    vec3 normal   = mix(normalSW, normalNE, alpha.x);

    // calculate height above surface
    // VP_demAverageHeight is substracted in order to increase the accuracy on high mountains and in deep valleys
    float height = length(VP_matModelView[0]) * VP_heightScale 
                   * (VP_getVertexHeight(iPosition) - VP_demAverageHeight);

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

        vec3 center = (VP_matModelView * vec4(0, 0, 0, 1)).xyz;
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
    // neighbour vertices (p: positive direction, n: negative direction)
    ivec2 pp = ivec2(iPosition.x + 1, iPosition.y);
    ivec2 nn = ivec2(iPosition.x - 1, iPosition.y);
    ivec2 np = ivec2(iPosition.x, iPosition.y + 1);
    ivec2 pn = ivec2(iPosition.x, iPosition.y - 1);

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
    // neighbour vertices (px: x direction, py: y direction)
    ivec2 px = ivec2(iPosition.x + 1, iPosition.y);
    ivec2 py = ivec2(iPosition.x, iPosition.y + 1);

    // euclidian position of neighbour vertices
    vec3 p_px = VP_getVertexPosition(px, mode);
    vec3 p_py = VP_getVertexPosition(py, mode);

    // central differences as approximation for surface tangents
    vec3 dx = p_px - centerPos;
    vec3 dy = p_py - centerPos;

    // cross product of tangents -> normal
    return normalize(cross(dx, dy));
}
