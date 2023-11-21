#version 440

uniform sampler3D uTexture;

uniform mat4      uMatInvMV;
uniform mat4      uMatInvP;

uniform vec2      uLonRange;
uniform vec2      uLatRange;
uniform vec2      uHeightRange;

uniform vec3      uInnerRadii;
uniform vec3      uOuterRadii;

uniform vec3      uBodyRadii;

in vec3 rayDirection;

out vec4 FragColor;

const float PI = 3.14159265359;

struct Ray {
  vec3 origin;
  vec3 direction;
};

bool raySphereIntersect(Ray ray, float radius, out float entry, out float exit) {
  float b   = dot(ray.origin, ray.direction);
  float c   = dot(ray.origin, ray.origin) - radius * radius;
  float det = b * b - c;

  if (det < 0.0) {
    return false;
  }

  det = sqrt(det);

  float d1 = -b - det;
  float d2 = -b + det;

  if (d1 < 0 && d2 < 0) {
    return false;
  }

  entry = d1;
  exit = d2;

  return true;
}

vec3 getLngLatHeight(vec3 position) {
  vec3 result;

  if (position.z != 0.0) {
    result.x = atan(position.x / position.z);

    if (position.z < 0 && position.x < 0) {
      result.x -= PI;
    }

    if (position.z < 0 && position.x >= 0) {
      result.x += PI;
    }

  } else if (position.x == 0) {
    result.x = 0.0;
  } else if (position.x < 0) {
    result.x = -PI * 0.5;
  } else {
    result.x = PI * 0.5;
  }

  result.y = asin(position.y / length(position));
  // result.xy *= 180.0 / PI;

  result.z = length(position) - uBodyRadii.x;

  return result;
}

bool pointInsideVolume(vec3 lngLatHeight) {
  float lon    = lngLatHeight.x;
  float lat    = lngLatHeight.y;
  float height = lngLatHeight.z;
  return lon > uLonRange.x
    && lon < uLonRange.y
    && lat > uLatRange.x
    && lat < uLatRange.y
    && height > uHeightRange.x
    && height < uHeightRange.y
  ;
}

vec4 getColor(vec3 lngLatHeight) {
  float lon    = lngLatHeight.x;
  float lat    = lngLatHeight.y;
  float height = lngLatHeight.z;
  return texture(uTexture, vec3(
      (lon - uLonRange.x) / (uLonRange.y - uLonRange.x),
      (lat - uLatRange.x) / (uLatRange.y - uLatRange.x),
      (height - uHeightRange.x) / (uHeightRange.y - uHeightRange.x)
  ));
}

void main() {
  Ray r = Ray(uMatInvMV[3].xyz, normalize(rayDirection));

  float outerEntry;
  float outerExit;
  if (raySphereIntersect(r, uOuterRadii.x, outerEntry, outerExit)) {

    float innerExit;
    float innerEntry;
    bool intersectsInner = raySphereIntersect(r, uInnerRadii.x, innerEntry, innerExit);

    float entry;
    float exit;
    if (!intersectsInner) {
      entry = outerEntry;
      exit = outerExit;
    } else if (innerEntry > 0) {
      entry = outerEntry;
      exit = innerEntry;
    } else if (outerEntry < 0 && innerEntry < 0 && !raySphereIntersect(r, uBodyRadii.x, entry, exit)) {
      entry = innerExit;
      exit = outerExit;
    } else {
      discard;
    }

    const float stepLength = 200.0;
    entry = max(stepLength, entry);

    vec4 finalColor = vec4(0.0, 0.0, 0.0, 0.0);

    float distance = exit - entry;
    vec3 entryPoint = r.origin + (r.direction * (entry + stepLength));
    vec3 llhEntry = getLngLatHeight(entryPoint);

    if (pointInsideVolume(llhEntry)) {
      for (float travelled = stepLength; travelled < distance && finalColor.a < 1.0; travelled += stepLength) {
        vec3 p = entryPoint + (r.direction * travelled);
        vec3 llh = getLngLatHeight(p);
        vec4 currentColor = getColor(llh);
        float currentAlpha = currentColor.a;

        currentColor.rgb *= currentAlpha;
        finalColor = (1.0 - finalColor.a) * currentColor + finalColor;
      }

      FragColor = finalColor;
    } else {
      discard;
    }
  } else {
    discard;
  }
}