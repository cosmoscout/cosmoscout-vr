#version 330

#define SHOW_TEXTURE_RGB  0

uniform float heightMin;
uniform float heightMax;
uniform float slopeMin;
uniform float slopeMax;
uniform float ambientBrightness;
uniform float texGamma;
uniform float farClip;
uniform vec4 uSunDirIlluminance;

uniform sampler1D heightTex;
uniform sampler2D fontTex;

// ==========================================================================
// include helper functions/declarations from VistaPlanet
$VP_TERRAIN_SHADER_UNIFORMS
$VP_TERRAIN_SHADER_FUNCTIONS

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
} fsIn;

vec3 heat(float v) {
  float value = 1.0-v;
  return (0.5+0.5*smoothstep(0.0, 0.1, value))*vec3(
         smoothstep(0.5, 0.3, value),
      value < 0.3 ?
         smoothstep(0.0, 0.3, value) :
         smoothstep(1.0, 0.6, value),
         smoothstep(0.4, 0.6, value)
  );
}


// outputs
// ==========================================================================

layout(location = 0) out vec3 fragColor;

vec3 SRGBtoLINEAR(vec3 srgbIn)
{
  vec3 bLess = step(vec3(0.04045),srgbIn);
  return mix( srgbIn/vec3(12.92), pow((srgbIn+vec3(0.055))/vec3(1.055),vec3(2.4)), bLess );
}

void main()
{
  if (VP_shadowMapMode)
  {
    gl_FragDepth = gl_FragCoord.z;
    return;
  }

  gl_FragDepth = length(fsIn.position) / farClip;

  fragColor = vec3(1);

  vec3 idealNormal = normalize(fsIn.position - fsIn.planetCenter);

  #if $LIGHTING_QUALITY > 1
    vec3 surfaceNormal = normalize(fsIn.normal);
  #else
    vec3 dx = dFdx(fsIn.position);
    vec3 dy = dFdy(fsIn.position);
    vec3 surfaceNormal = normalize(cross(dx, dy));
  #endif

  #if $SHOW_TEXTURE
    #if $TEXTURE_IS_RGB
      fragColor = texture(VP_texIMG, vec3(fsIn.texcoords, VP_layerIMG)).rgb;
    #else
      fragColor = texture(VP_texIMG, vec3(fsIn.texcoords, VP_layerIMG)).rrr;
    #endif
    
    #if $ENABLE_HDR
      fragColor = SRGBtoLINEAR(fragColor);
    #endif

    fragColor = pow(fragColor, vec3(1.0 / texGamma));
  #endif

  #if $COLOR_MAPPING_TYPE==1
    if (/*fragColor == vec3(1) ||*/ $SHOW_TEXTURE != 1 || $MIX_COLORS == 1) {
      // map height to color scale
      float height      = clamp(fsIn.height, heightMin, heightMax);
      float height_norm = (height - heightMin) / (heightMax - heightMin);
      fragColor *= texture(heightTex, height_norm).rgb;
    }
  #endif

  #if $COLOR_MAPPING_TYPE==2
    if (fragColor == vec3(1) || $MIX_COLORS == 1) {
      float slope = acos(dot(idealNormal, surfaceNormal));
      float fac = clamp((slope - slopeMin)/(slopeMax - slopeMin), 0.0, 1.0);
      fragColor *= texture(heightTex, fac).rgb;
    }
  #endif

  fragColor = fragColor * uSunDirIlluminance.w;

  float directLight = 1.0;
  float ambientLight = ambientBrightness;

  #if $ENABLE_SHADOWS
    directLight *= VP_getShadow(fsIn.position);
  #endif

  #if $ENABLE_LIGHTING
    // hill shading / pseudo ambient occlusion
    const float hillShadingIntensity = 0.5;
    ambientLight *= mix(1.0, max(0, dot(idealNormal, surfaceNormal)), hillShadingIntensity);
    
    vec3 sunDir = normalize(fsIn.sunDir);
    directLight *= max(dot(surfaceNormal, sunDir), 0.0);
  #endif

  fragColor = mix(fragColor*ambientLight, fragColor, directLight);

  #if $SHOW_TILE_BORDER
    // color area by level
    const float minLevel = 1;
    const float maxLevel = 15;
    const float brightness = 0.5;

    float level = clamp(log2(float(VP_tileOffsetScale.z)), minLevel, maxLevel);
    vec4 debugColor = vec4(heat((level - minLevel)/(maxLevel-minLevel)), 0.5);
    debugColor.rgb = mix(debugColor.rgb, vec3(1), brightness);

    // create border pixel row color
    vec2 demPosition = fsIn.vertexPosition + VP_demOffsetScale.xy;
    vec2 imgPosition = (fsIn.vertexPosition + VP_imgOffsetScale.xy) / VP_imgOffsetScale.z * VP_MAXVERTEX;
    float edgeWidth = 0.5;

    // make border between image patches gray
    if (imgPosition.x < edgeWidth || 
        imgPosition.y < edgeWidth ||
        imgPosition.x > VP_MAXVERTEX-edgeWidth || 
        imgPosition.y > VP_MAXVERTEX-edgeWidth)
    {
        debugColor = vec4(0.0, 0.0, 0.0, 0.5);
    }

    const vec3 neighbourHigher = vec3(0, 0.3, 0);
    const vec3 neighbourLower  = vec3(0.3, 0, 0);
    const vec3 neighbourSame   = vec3(0);

    // make border between dem patches colorful, based on the adjacent levels
    if (demPosition.x < edgeWidth) {
        if      (VP_edgeDelta.z < 0) debugColor = vec4(neighbourHigher, 1.0);
        else if (VP_edgeDelta.z > 0) debugColor = vec4(neighbourLower,  1.0);
        else                         debugColor = vec4(neighbourSame,   1.0);
    } else if (demPosition.y < edgeWidth) {
        if      (VP_edgeDelta.w < 0) debugColor = vec4(neighbourHigher, 1.0);
        else if (VP_edgeDelta.w > 0) debugColor = vec4(neighbourLower,  1.0);
        else                         debugColor = vec4(neighbourSame,   1.0);
    } else if (demPosition.x > VP_MAXVERTEX-edgeWidth) {
        if      (VP_edgeDelta.x < 0) debugColor = vec4(neighbourHigher, 1.0);
        else if (VP_edgeDelta.x > 0) debugColor = vec4(neighbourLower,  1.0);
        else                         debugColor = vec4(neighbourSame,   1.0);
    } else if (demPosition.y > VP_MAXVERTEX-edgeWidth) {
        if      (VP_edgeDelta.y < 0) debugColor = vec4(neighbourHigher, 1.0);
        else if (VP_edgeDelta.y > 0) debugColor = vec4(neighbourLower,  1.0);
        else                         debugColor = vec4(neighbourSame,   1.0);
    }

    fragColor = mix(fragColor, debugColor.rgb, debugColor.a);
  #endif
  
  #if $ENABLE_SHADOWS_DEBUG && $ENABLE_SHADOWS
    float cascade = VP_getCascade(fsIn.position);
    if (cascade >= 0)
    {
        fragColor = mix(fragColor, heat(1-cascade/(VP_shadowCascades-1)), 0.2);
    }
  #endif

  vec3  viewDir = -fsIn.position;
  float camDist = length(viewDir);
  float centerDist = length(fsIn.planetCenter);

  #if $SHOW_LAT_LONG || $SHOW_LAT_LONG_LABELS
  {
    #if $ENABLE_LIGHTING
      float fIdealLightIntensity = dot(idealNormal, sunDir) * (1.0 - ambientLight) + ambientLight;
      vec3 grid_color = mix(fragColor, vec3(mix(1.0, 0.0, clamp(fIdealLightIntensity + 1.0, 0.0, 1.0))), 0.8);
    #else
      vec3 grid_color = mix(fragColor, vec3(0), 0.8);
    #endif

    const float spacings[4]        = float[](500.0, 50.0, 5.0, 1.0);
    const float distances[4]       = float[](1.5, 0.8, 0.4, 0.1);
    const float intensities[4]     = float[](0.5, 0.4, 0.3, 0.2);
    const float label_distances[3] = float[](0.8, 0.4, 0.1);

    for (int i=0; i<4; ++i) {
      float grid_fade = (1.0 - clamp(camDist/centerDist/distances[i],   0.0, 1.0));

      float latDeg = (fsIn.lngLat.y / VP_PI) * 9000.0;
      float latDh = mod(latDeg, spacings[i]);
      float latDhMirrored = (latDh > 0.5 * spacings[i]) ? spacings[i] - latDh : latDh;

      float wLatDeg = fwidth(latDeg);
      float latA = clamp(abs(latDhMirrored / wLatDeg), 0.0, 1.0);

      #if $SHOW_LAT_LONG
        fragColor = mix(fragColor, grid_color, (1.0 - latA)*grid_fade*intensities[i]);
      #endif

      float lngDeg = (fsIn.lngLat.x / VP_PI) * 9000.0;
      float lngDh = mod(lngDeg, spacings[i]);
      float lngDhMirrored = (lngDh > 0.5 * spacings[i]) ? spacings[i] - lngDh : lngDh;

      float wLngDeg = fwidth(lngDeg);
      float lngA = clamp(abs(lngDhMirrored / wLngDeg), 0.0, 1.0);

      #if $SHOW_LAT_LONG
        fragColor = mix(fragColor, grid_color, (1.0 - lngA)*grid_fade*intensities[i]);
      #endif

      // print labels
      #if $SHOW_LAT_LONG_LABELS
        if (i < 3) {
          float curr_label_fade =         (1.0 - clamp(camDist/centerDist/label_distances[i]-0.1,   0.0, 1.0));
          float next_label_fade = i < 2 ? (1.0 - clamp(camDist/centerDist/label_distances[i+1]-0.1, 0.0, 1.0)) : 0.0;

          // divide current cell in sub cells, 2 rows and 7 columns
          const int scaleX = 30;
          float scaleY = 15 / (1 - abs(fsIn.lngLat.y / VP_PI));
          // float scaleY = 10 * abs(asin(idealNormal.y));

          vec2 rowCol = vec2(lngDh * scaleX, latDh * scaleY) / spacings[i];
          if (rowCol.x > scaleX-7 && rowCol.x < scaleX && rowCol.y > 0 && rowCol.y < 2) {

            // compute mipmap level
            vec2 tcdx = dFdx(rowCol/scaleX);
            vec2 tcdy = dFdy(rowCol/scaleY);

            // get texture coordinate per digit and flip horizontally
            vec2 digitCoord = rowCol - ivec2(rowCol);

            float number, digit, suffix;

            if (rowCol.y < 1) {
              if (latDeg >= 0) {
                suffix = 10;
                number = 10 * (latDeg - latDh);
              } else {
                suffix = 11;
                number = -10 * (latDeg - latDh);
              }
            } else {
              if (lngDeg > 0) {
                suffix = 12;
                number = 10 * (lngDeg - lngDh + spacings[i]);
              } else {
                suffix = 13;
                number = -10 * (lngDeg - lngDh + spacings[i]);
              }
            }

            int numberInt = int(number/50 + 0.5 * sign(number));

            switch (int(rowCol.x)) { // char column
              case scaleX - 1:
                digit = suffix;
                break;
              case scaleX - 2:
                digit = 15; // '°'
                break;
              case scaleX - 3:
                digit = mod(numberInt, 10);
                break;
              case scaleX - 4:
                digit = 14; // '.'
                break;
              case scaleX - 5:
                digit = mod((numberInt / 10), 10);
                break;
              case scaleX - 6:
                digit = numberInt >= 100 ? mod((numberInt / 100), 10) : -1;
                break;
              case scaleX - 7:
                digit = numberInt >= 1000 ? mod((numberInt / 1000), 10) : -1;
                break;
              default:
                digit = -1;
                break;
            }

            if (digit >= 0) {
              const float totalDigits = 16;
              digitCoord.x = (digitCoord.x + digit)/totalDigits;

              float fade = 1.0 - textureGrad(fontTex, digitCoord, tcdx, tcdy).r;
              fade *= max(dot(viewDir/camDist, idealNormal), 0.0);
              fade *= clamp(curr_label_fade - next_label_fade, 0.0, 1.0);

              fragColor = mix(fragColor, grid_color, fade);
            }

          }
        }
      #endif
    }
  }
  #endif


  #if $SHOW_HEIGHT_LINES
  {
    const float levels[4]      = float[](5000.0, 1000.0, 100.0, 10.0);
    const float distances[4]   = float[](1.5, 1.0, 0.5, 0.1);
    const float intensities[4] = float[](0.5, 0.4, 0.3, 0.2);

    float wh = fwidth(fsIn.height);

    for (int i=0; i<4; ++i) {
      float fade = (1.0 - clamp(camDist/centerDist/distances[i], 0.0, 1.0)) * intensities[i];

      // distance from IsoLevel
      float levelDh = mod(fsIn.height, levels[i]);
      float levelDhMirrored = (levelDh > 0.5 * levels[i]) ? levels[i] - levelDh : levelDh;

      float level = clamp(abs(levelDhMirrored / wh), 0.0, 1.0);
      fragColor = mix(fragColor, (1.0 - intensities[i]) * fragColor, (1.0 - level)*fade);

      // layer shading
      float layer = clamp(abs((levels[i] - levelDh) / wh) * 0.1, 0.0, 1.0);
      fragColor = mix(fragColor, (1.0 - intensities[i]) * fragColor, (1.0 - layer)*fade*0.5);
    }
  }
  #endif
}
