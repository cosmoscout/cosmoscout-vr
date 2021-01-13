////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

vec3 VP_getShadowMapCoords(int cascade, vec3 position)
{
    vec4 smap_coords = VP_shadowProjectionViewMatrices[cascade] * vec4(position, 1.0);
    return (smap_coords.xyz / smap_coords.w) * 0.5 + 0.5;
}

int VP_getCascade(vec3 position)
{
    for (int i=0; i<VP_shadowCascades; ++i)
    {
        vec3 coords = VP_getShadowMapCoords(i, position);

        if (coords.x > 0 && coords.x < 1 && 
            coords.y > 0 && coords.y < 1 &&
            coords.z > 0 && coords.z < 1)
        {
            return i;
        }
    }

    return -1;
} 

float VP_getShadow(vec3 position)
{
    int cascade = VP_getCascade(position);

    if (cascade < 0)
    {
        return 1.0;
    }

    vec3 coords = VP_getShadowMapCoords(cascade, position);

    float shadow = 0;
    float size = 0.001;

    for(int x=-1; x<=1; x++){
        for(int y=-1; y<=1; y++){
            vec2 off = vec2(x,y)*size;

            // Dynamic array lookups are not supported in OpenGL 3.3
            if      (cascade == 0) shadow += texture(VP_shadowMaps[0], coords - vec3(off, VP_shadowBias * 1));
            else if (cascade == 1) shadow += texture(VP_shadowMaps[1], coords - vec3(off, VP_shadowBias * 2));
            else if (cascade == 2) shadow += texture(VP_shadowMaps[2], coords - vec3(off, VP_shadowBias * 3));
            else if (cascade == 3) shadow += texture(VP_shadowMaps[3], coords - vec3(off, VP_shadowBias * 4));
            else                   shadow += texture(VP_shadowMaps[4], coords - vec3(off, VP_shadowBias * 5));
        }
    }

    return shadow / 9.0;
}

