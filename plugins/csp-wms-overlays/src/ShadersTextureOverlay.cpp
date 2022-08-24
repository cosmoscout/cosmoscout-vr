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

    uniform dvec2         uLonRange;
    uniform dvec2         uLatRange;
    uniform vec3          uRadii;

    uniform float         uAmbientBrightness;
    uniform float         uSunIlluminance;
    uniform vec3          uSunDirection;

    in vec2 texcoord;

    const float PI = 3.14159265359;

    // ===========================================================================
    vec3 GetPosition(float fDepth)
    {
        vec4  posMS = uMatInvMVP * vec4(2.0 * texcoord - 1.0, fDepth*2.0 - 1.0 , 1.0);
        return posMS.xyz / posMS.w;
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
        vec2  vTexcoords = texcoord*textureSize(uDepthBuffer);
        float fDepth     = texture(uDepthBuffer, vTexcoords).r;

        if (fDepth == 1.0) 
        {
            discard;
        }else{
            vec3 worldPos = GetPosition(fDepth);
            vec2 lnglat   = surfaceToLngLat(worldPos, uRadii);

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
                    result = SRGBtoLINEAR(result) * uSunIlluminance / PI;
                #else
                    result = result * uSunIlluminance;
                #endif

                #ifdef ENABLE_LIGHTING
                    //Lighting using a normal calculated from partial derivative
                    vec3 dx = dFdx( worldPos );
                    vec3 dy = dFdy( worldPos );

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
