////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_TREE_BVH_HPP
#define CSP_TREE_BVH_HPP

#include <glm/glm.hpp>
#include <memory>
#include <algorithm>
#include <vector>

#include "../src/Atmosphere.hpp"

const float DEFAULT_DENSITY_CUTOFF = 1.0e-2f;

static unsigned int GetIndexFromPos(glm::vec3 pos, glm::uvec3 &dimensions);
static glm::vec3 GetPosFromIndex(unsigned int index, glm::uvec3 &dimensions);

struct TreeNode {
    // Index-based retrieval (3D noise texture stored as vector<float>)
    glm::vec3 aabbMin, aabbMax;
    // Octree children-count is static (=8), so either firstChildIndex or density is occupied
    unsigned int firstChildIndex; // firstChildIndex > 0 => branch node
    float val; // firstChildIndex == 0 => leaf node, so val is well-defined

    TreeNode() {
        aabbMin = glm::vec3(0.0);
        aabbMax = glm::vec3(0.0);
        firstChildIndex = 0;
        val = -0.0f;
    }

    bool IsLeaf() const {
        return firstChildIndex == 0;
    }

    bool HitRay(glm::vec3 origin, glm::vec3 dir, float densityCutoff = DEFAULT_DENSITY_CUTOFF) {
        throw;
    }
};

struct CloudProperties {
    csp::atmospheres::Uniforms uniforms;
    const float *noise, *noise2d, *cloud, *cloudType;
    glm::uvec3 noiseDim, noise2dDim, cloudDim, cloudTypeDim;

    // CloudProperties(csp::atmospheres::Uniforms uniforms, std::vector<float> noise, std::vector<float> noise2d) {
    //     this->uniforms = uniforms;
    //     this->noise = noise;
    //     this->noise2d = noise2d;
    // }
};

class Tree {
private:
    glm::uvec3 dimensions;
    unsigned int maxDepth, maxNodeCount, usedNodeIndex;
    CloudProperties properties;
    std::unique_ptr<TreeNode[]> nodes;

    // Calculates cloud density at the given index.
    float GetDensity(glm::vec3 pos);

    void Subdivide(unsigned int index);
    void UpdateBounds(unsigned int index, unsigned int relChildIndex);

public:
    Tree(glm::uvec3 dimensions, unsigned int maxDepth, CloudProperties properties);
    void Build();
};

static unsigned int GetIndexFromPos(glm::vec2 pos, const glm::uvec2 &dimensions) {
    return (int)pos.x * dimensions.x + (int)pos.y * dimensions.y;
}

static unsigned int GetIndexFrom3DPos(glm::vec3 pos, const glm::uvec3 &dimensions) {
    return (int)pos.x * dimensions.x + (int)pos.y * dimensions.y + (int)pos.z * dimensions.z;
}

static glm::vec3 GetPosFromIndex(unsigned int index, const glm::uvec3 &dimensions) {
    glm::uvec3 pos;
    pos.x = index % dimensions.x;
    pos.y = int(index / dimensions.x) % dimensions.y;
    pos.z = int(index / (dimensions.x * dimensions.y));
    return pos;
}

static glm::vec4 GetTexture(const float *data2d, glm::uvec2 &&dimensions, glm::vec2 texCoords) {
    unsigned int index = GetIndexFromPos(texCoords, dimensions);
    glm::vec4 texel(0.0);
    for (int i = 0; i < 3; i++) {
        texel[i] = data2d[index + i];
    }
    return texel;
}

static glm::vec4 GetTexture3D(const float *data, glm::uvec3 &&dimensions, glm::vec3 texCoords) {
    unsigned int index = GetIndexFrom3DPos(texCoords, dimensions);
    glm::vec4 texel(0.0);
    for (int i = 0; i < 3; i++) {
        texel[i] = data[index + i];
    }
    return texel;
}

static glm::vec2 GetSphericalCoords(glm::vec3 pos) {
    glm::vec2 result;
    result.x = atan2(pos.x, pos.z);
    result.y = asin(pos.y / length(pos));
    return result;
}

static float Remap(float t, float oldMin, float oldMax, float newMin, float newMax) {
    float tScaled = t / (oldMax - oldMin);
    return std::clamp(tScaled * (newMax - newMin), std::min(newMin, newMax), std::max(newMin, newMax));
}

// Constants from csp-atmosphere.frag (must be updated when corresponding constants in shader file are changed!)
const float CUMULONIMBUS_START_HEIGHT = 1500;
const float CUMULONIMBUS_END_HEIGHT = 5000;
const float CLOUD_BASE_FRACTION = 0.;

const float CLOUD_TYPE_NOISE_WORLEY_SCALE = 5.3f;
const float CLOUD_TYPE_NOISE_PERLIN_SCALE = 30;

static glm::vec4 GetLocalCloudType(glm::vec2 texCoords, CloudProperties &properties){
      glm::vec4 worleySample = GetTexture(properties.noise2d, properties.noise2dDim, texCoords * CLOUD_TYPE_NOISE_WORLEY_SCALE);
      glm::vec4 perlinSample = GetTexture(properties.noise2d, properties.noise2dDim, texCoords * CLOUD_TYPE_NOISE_PERLIN_SCALE);
      float worleyNoise = worleySample.b;
      float perlinNoise = perlinSample.g;
      float cloudType = worleyNoise * 0.5f + perlinNoise * .5f;
      return glm::vec4(Remap(pow(cloudType, (float)properties.uniforms.cloudTypeExponent),
        (float)properties.uniforms.cloudRangeMin, (float)properties.uniforms.cloudRangeMax, (float)properties.uniforms.cloudTypeMin,
        (float)properties.uniforms.cloudTypeMax), perlinSample.y, perlinSample.z, perlinSample.w);
}

#endif