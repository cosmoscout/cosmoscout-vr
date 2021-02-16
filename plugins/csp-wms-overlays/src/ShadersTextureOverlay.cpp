#include "TextureOverlayRenderer.hpp"

#include <string>

namespace csp::wmsoverlays {

const std::string TextureOverlayRenderer::SURFACE_GEOM = R"(
    #version 330 core

    layout(points) in;
    layout(triangle_strip, max_vertices = 4) out;

    out vec2 texcoord;

    void main() 
    {
        gl_Position = vec4( 1.0, 1.0, 0.5, 1.0 );
        texcoord = vec2( 1.0, 1.0 );
        EmitVertex();

        gl_Position = vec4(-1.0, 1.0, 0.5, 1.0 );
        texcoord = vec2( 0.0, 1.0 ); 
        EmitVertex();

        gl_Position = vec4( 1.0,-1.0, 0.5, 1.0 );
        texcoord = vec2( 1.0, 0.0 ); 
        EmitVertex();

        gl_Position = vec4(-1.0,-1.0, 0.5, 1.0 );
        texcoord = vec2( 0.0, 0.0 ); 
        EmitVertex();

        EndPrimitive(); 
    }
)";

const std::string TextureOverlayRenderer::SURFACE_VERT = R"(
    #version 330 core

    void main()
    {
    }
)";

const std::string TextureOverlayRenderer::SURFACE_FRAG = R"(
    out vec4 FragColor;

    uniform sampler2DRect uDepthBuffer;
    uniform sampler2D     uFirstTexture;
    uniform sampler2D     uSecondTexture;

    uniform float         uFade;
    uniform bool          uUseFirstTexture;
    uniform bool          uUseSecondTexture;

    uniform mat4          uMatInvMVP;
    uniform dmat4         uMatInvMV;
    uniform mat4          uMatInvP;
    uniform mat4          uMatMV;

    uniform float         uFarClip;
    uniform dvec2         uLonRange;
    uniform dvec2         uLatRange;
    uniform vec3          uRadii;

    uniform bool          uEnableLighting;
    uniform float         uAmbientBrightness;
    uniform float         uSunIlluminance;
    uniform vec3          uSunDirection;

    in vec2 texcoord;

    const float PI = 3.14159265359;

    // ===========================================================================
    float GetDepth()
    {
        vec2  vTexcoords = texcoord*textureSize(uDepthBuffer);
        float fDepth     = texture(uDepthBuffer, vTexcoords).r;


        // We need to return a distance which is guaranteed to be larger
        // than the largest ray length possible. As the atmosphere has a
        // radius of 1.0, 1000000 is more than enough.
        if (fDepth == 1) return 1000000.0;

        float linearDepth = fDepth * uFarClip;
        vec4 posFarPlane = uMatInvP * vec4(2.0*texcoord-1, 1.0, 1.0);
        vec3 posVS = normalize(posFarPlane.xyz) * linearDepth;

        float distance = length(float(uMatInvMV[3].xyz - (uMatInvMV * vec4(posVS, 1.0)).xyz));
        return distance;
    }

    // ===========================================================================
    dvec3 GetPosition()
    {
        vec2  vTexcoords = texcoord*textureSize(uDepthBuffer);
        float fDepth     = texture(uDepthBuffer, vTexcoords).r;

        float  linearDepth = fDepth * uFarClip;
        dvec4  posFar = uMatInvP * dvec4(2.0 * texcoord - 1, 1.0 , 1.0);
        dvec3  posVS = normalize(posFar.xyz) * linearDepth;
        dvec4  posWorld = uMatInvMV * dvec4(posVS, 1.0);

        return posWorld.xyz;
    }

    // ===========================================================================
    dvec2 GetLngLat(dvec3 vPosition)
    {
        dvec2 result = dvec2(-2);

        if (vPosition.z != 0.0)
        {
            //result.x = atan2(vPosition.z, vPosition.x);
            result.x = atan(float(vPosition.x / vPosition.z));

            if (vPosition.z < 0 && vPosition.x < 0)
                result.x -= PI;
            if (vPosition.z < 0 && vPosition.x >= 0)
                result.x += PI;
        }
        else if (vPosition.x == 0)
            result.x = 0.0;
        else if (vPosition.x < 0)
            result.x = -PI * 0.5;
        else
            result.x = PI * 0.5;

        // geocentric latitude of the input point
        result.y = float(asin(float(vPosition.y / length(vPosition))));

        return result;
    }

    // ===========================================================================
    vec3 scaleToGeodeticSurface(vec3 cartesian, vec3 radii) {
        vec3 radii2        = radii * radii;
        vec3 radii4        = radii2 * radii2;
        vec3 oneOverRadii2 = 1.0 / radii2;
        vec3 cartesian2    = cartesian * cartesian;

        float beta  = 1.0 / sqrt(dot(cartesian2, oneOverRadii2));
        float n     = length(beta * cartesian * oneOverRadii2);
        float alpha = (1.0 - beta) * (length(cartesian) / n);
        float s     = 0.0;
        float dSdA  = 1.0;

        vec3 d;

        do {
            alpha -= (s / dSdA);

            d    = vec3(1.0) + (alpha * oneOverRadii2);
            s    = dot(cartesian2, 1.0 / (radii2 * d * d)) - 1.0;
            dSdA = dot(cartesian2, 1.0 / (radii4 * d * d * d)) * -2.0;

        } while (abs(s) > 0.00000000001);

        return cartesian / d;
    }

    // ===========================================================================
    vec3 surfaceToNormal(vec3 cartesian, vec3 radii) {
        vec3 radii2        = radii * radii;
        vec3 oneOverRadii2 = 1.0 / radii2;
        return normalize(cartesian * oneOverRadii2);
    }

    // ===========================================================================
    vec2 surfaceToLngLat(vec3 cartesian, vec3 radii) {
        vec3 geodeticNormal = surfaceToNormal(cartesian, radii);
        return vec2(atan(geodeticNormal.x, geodeticNormal.z), asin(geodeticNormal.y));
    }

    // ===========================================================================
    vec3 SRGBtoLINEAR(vec3 srgbIn)
    {
        vec3 bLess = step(vec3(0.04045),srgbIn);
        return mix( srgbIn/vec3(12.92), pow((srgbIn+vec3(0.055))/vec3(1.055),vec3(2.4)), bLess );
    }

    // ===========================================================================
    void main()
    {     
        float fDepth = GetDepth();
        if (fDepth == 1000000.0) 
        {
            discard;
        }else{
            dvec3 worldPos = GetPosition();
            vec2 lnglat    = surfaceToLngLat(vec3(worldPos.x, worldPos.y, worldPos.z), uRadii);

            FragColor = vec4(worldPos, 1.0);

            if(lnglat.x > uLonRange.x && lnglat.x < uLonRange.y &&
               lnglat.y > uLatRange.x && lnglat.y < uLatRange.y)
            {
                double norm_u = (lnglat.x - uLonRange.x) / (uLonRange.y - uLonRange.x);
                double norm_v = (lnglat.y - uLatRange.x) / (uLatRange.y - uLatRange.x);
                vec2 newCoords = vec2(float(norm_u), float(1.0 - norm_v));

                vec4 color = vec4(0.);
                if (uUseFirstTexture) {
                  color = texture(uFirstTexture, newCoords);

                  // Fade second texture in.
                  if(uUseSecondTexture) {
                    vec4 secColor = texture(uSecondTexture, newCoords);
                    color = mix(secColor, color, uFade);
                  }
                }

                vec3 result = color.rgb;

                #ifdef ENABLE_HDR
                    result = SRGBtoLINEAR(result);
                #endif

                result = result * uSunIlluminance;

                #ifdef ENABLE_LIGHTING
                    //Lighting using a normal calculated from partial derivative
                    vec3  fPos    = vec3(worldPos); //cast from double to float
                    vec3  dx      = dFdx( fPos );
                    vec3  dy      = dFdy( fPos );

                    vec3 N = normalize(cross(dx, dy));
                    //N *= sign(N.z);
                    float NdotL = max(dot(N, uSunDirection), 0.);

                    result = mix(result * uAmbientBrightness, result, NdotL);
                #endif
               
                FragColor = vec4(result, color.a);
            }
            else
                discard;
        }
    }
)";

} // namespace csp::wmsoverlays
