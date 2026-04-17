////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_TREE_BVH_HPP
#define CSP_TREE_BVH_HPP

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

#include <memory>
#include <algorithm>
#include <vector>
#include <math.h>
#include "VistaBase/VistaStreamUtils.h"

#include "utils.hpp"

namespace csp::atmospheres {
    const float DEFAULT_DENSITY_CUTOFF = 1.0e-2f;

    static unsigned int GetIndexFromPos(glm::vec3 pos, glm::uvec3 &dimensions);
    static glm::vec3 GetPosFromIndex(unsigned int index, glm::uvec3 &dimensions);

    struct TreeNode {
        // Index-based retrieval (3D noise texture stored as vector<float>)
        glm::vec3 aabbMin, aabbMax;
        // Octree children-count is static (=8), so either firstChildIndex or density is occupied
        unsigned int firstChildIndex; // firstChildIndex > 0 => branch node
        float density; // firstChildIndex == 0 => leaf node, so val is well-defined

        TreeNode() {
            aabbMin = glm::vec3(0.0);
            aabbMax = glm::vec3(0.0);
            firstChildIndex = 0;
            density = -0.0f;
        }

        glm::vec3 GetExtends() const {
            return aabbMax - aabbMin;
        }

        bool IsLeaf() const {
            return firstChildIndex == 0;
        }

        bool HitRay(glm::vec3 origin, glm::vec3 dir, float densityCutoff = DEFAULT_DENSITY_CUTOFF) {
            throw;
        }
    };

    struct CloudRenderSettings {
        float cloudQuality = 1.0f;
        float cloudTypeExponent = 1.0f;
        float cloudRangeMin = 0.0f;
        float cloudRangeMax = 1.0f;
        float cloudTypeMin = 0.0f;
        float cloudTypeMax = 1.0f;
        float cloudDensityMultiplier = 1.0f;
        float cloudAbsorption = 0.0f;
        float cloudCoverageExponent = 1.0f;
        float cloudCutoff = 0.1f;
        float cloudLFRepetitionScale = 5000.0f;
        float cloudHFRepetitionScale = 1231.0f;
    };

    struct CloudProperties {
        CloudRenderSettings renderSettings;
        float planetRadius, cloudLayerHeight;
        float *noise, *noise2d;
        std::vector<float> cloud, cloudType;
        glm::uvec3 noiseDim;
        glm::uvec2 noise2dDim, cloudDim, cloudTypeDim;
    };

    class Tree {
    private:
        unsigned int maxDepth, maxNodeCount, usedNodeIndex;
        CloudProperties properties;
        std::unique_ptr<TreeNode[]> nodes;

        // Calculates cloud density at the given index.
        float GetDensity(glm::vec3 pos);
        // Calculates average density throughout the node (basically the cost function for decision to subdivide).
        float GetTotalDensity(unsigned int index, unsigned int totalSamples);
        void Subdivide(unsigned int index, unsigned int depth);
        void UpdateBounds(unsigned int index, unsigned int relChildIndex);

    public:
        Tree(glm::vec3 totalBoundsMin, glm::vec3 totalBoundsMax, unsigned int maxDepth, CloudProperties properties);
        void Build();

        TreeNode *GetNodes() const {
            return &nodes[0];
        }

        unsigned int GetUsedNodeCount() const {
            return usedNodeIndex + 1; // all used node indices
        }
    };

    static glm::vec2 RollOverVector(glm::vec2 pos, const glm::uvec2 &dimensions) {
        glm::ivec2 intPart(0);
        intPart.x = (int)pos.x;
        intPart.y = (int)pos.y;

        glm::ivec2 intPartRolledOver;
        intPartRolledOver.x = intPart.x % dimensions.x;
        intPartRolledOver.y = intPart.y % dimensions.y;

        glm::vec2 decimalRemainder = (glm::vec2)intPart - pos;
        return (glm::vec2)intPartRolledOver + decimalRemainder;
    }

    static glm::vec3 RollOverVector3D(glm::vec3 pos, const glm::uvec3 &dimensions) {
        glm::ivec3 intPart(0);
        intPart.x = (int)round(pos.x);
        intPart.y = (int)round(pos.y);
        intPart.z = (int)round(pos.z);

        glm::ivec3 intPartRolledOver;
        intPartRolledOver.x = intPart.x % dimensions.x;
        intPartRolledOver.y = intPart.y % dimensions.y;
        intPartRolledOver.z = intPart.z % dimensions.z;

        glm::vec3 decimalRemainder = (glm::vec3)intPart - pos;
        return (glm::vec3)intPartRolledOver + decimalRemainder;
    }

    enum IndexingMode {
        Clamp,
        Repeat
    };

    static unsigned int GetIndexFromPos(glm::vec2 pos, const glm::uvec2 &dimensions, IndexingMode indexing) {
        // Simulate a GL_REPEAT texture.
        // When texture is read out of bounds, repeat the texture so that read position is inside texture.
        if (indexing == IndexingMode::Repeat)
            pos = RollOverVector(pos, dimensions);
        else if(indexing == IndexingMode::Clamp) {
            glm::vec2 newPos = pos = glm::clamp(pos, glm::vec2(0.0f), (glm::vec2)dimensions);
            // vstr::debug() << "Pos -> clampedPos (to " << glm::to_string(dimensions) << "): " << glm::to_string(pos)
            //     << " -> " << glm::to_string(newPos) << std::endl;
            pos = newPos;
        }
        return (int)pos.x + (int)pos.y * dimensions.x;
    }

    static unsigned int GetIndexFrom3DPos(glm::vec3 pos, const glm::uvec3 &dimensions, IndexingMode indexing) {
        if (indexing == IndexingMode::Repeat)
            pos = RollOverVector3D(pos, dimensions);
        else if(indexing == IndexingMode::Clamp)
            pos = glm::clamp(pos, glm::vec3(0.0f), (glm::vec3)dimensions);
        return (int)pos.x + (int)pos.y * dimensions.x + (int)pos.z * dimensions.x * dimensions.y;
    }

    static glm::vec3 Get3DPosFromIndex(unsigned int index, const glm::uvec3 &dimensions) {
        glm::uvec3 pos;
        pos.x = index % dimensions.x;
        pos.y = int(index / dimensions.x) % dimensions.y;
        pos.z = int(index / (dimensions.x * dimensions.y));
        return pos;
    }

    static glm::vec4 GetTexture(const float *data2d, glm::uvec2 &dimensions, glm::vec2 texCoords, unsigned int channels = 4, IndexingMode indexing = IndexingMode::Repeat) {
        // vstr::debug() << "GetTexture(clamped) texcoords = " << glm::to_string(texCoords) << std::endl;
        texCoords *= dimensions; // Tex coord input is normalised, so convert back to original dimensions.
        unsigned int index = GetIndexFromPos(texCoords, dimensions, indexing);
        // unsigned int size = dimensions.x * dimensions.y;
        // vstr::debug() << "Index = " << index << ", size = " << size << " (" << dimensions.x << ", " << dimensions.y << ")" << std::endl;

        glm::vec4 texel(0.0);
        for (unsigned int i = 0; i < channels; i++) {
            texel[i] = data2d[index + i];
        }
        return texel;
    }

    static glm::vec4 GetTexture3D(const float *data, glm::uvec3 &dimensions, glm::vec3 texCoords, unsigned int channels = 4, IndexingMode indexing = IndexingMode::Repeat) {
        texCoords *= dimensions; // Tex coord input is normalised, so convert back to original dimensions.
        unsigned int index = GetIndexFrom3DPos(texCoords, dimensions, indexing);
        // vstr::debug() << "Texture read: pos = " << glm::to_string(texCoords) << " -> index = " << index
        //     << ", dim = " << glm::to_string(dimensions) << ", size = " << dimensions.x * dimensions.y * dimensions.z << std::endl;
        // unsigned int size = dimensions.x * dimensions.y * dimensions.z;
        // vstr::debug() << "GetTexture3D index, size: " << index << ", " << size
        //     << "(" << dimensions.x << ", " << dimensions.y << ", " << dimensions.z << ")" << std::endl;

        glm::vec4 texel(0.0);
        for (unsigned int i = 0; i < channels; i++) {
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
        float tScaled = (t - oldMin) / (oldMax - oldMin);
        return std::clamp(newMin + tScaled * (newMax - newMin), std::min(newMin, newMax), std::max(newMin, newMax));
    }

    static float Mix(float a, float b, float t) {
        return a * (1 - t) + b * t;
    }

    // Constants from csp-atmosphere.frag (must be updated when corresponding constants in shader file are changed!)
    const float CUMULONIMBUS_START_HEIGHT = 1500;
    const float CUMULONIMBUS_END_HEIGHT = 5000;
    const float CLOUD_BASE_FRACTION = 0.;

    const float CLOUD_TYPE_NOISE_WORLEY_SCALE = 5.3f;
    const float CLOUD_TYPE_NOISE_PERLIN_SCALE = 30;

    static glm::vec4 GetLocalCloudType(glm::vec2 texCoords, CloudProperties &properties) {
        glm::vec4 worleySample = GetTexture(properties.noise2d, properties.noise2dDim, texCoords * CLOUD_TYPE_NOISE_WORLEY_SCALE);
        glm::vec4 perlinSample = GetTexture(properties.noise2d, properties.noise2dDim, texCoords * CLOUD_TYPE_NOISE_PERLIN_SCALE);
        // vstr::debug() << "Worley = " << glm::to_string(worleySample) << ", perlin = " << glm::to_string(perlinSample) << std::endl;
        float worleyNoise = worleySample.b;
        float perlinNoise = perlinSample.g;
        float cloudType = worleyNoise * 0.5f + perlinNoise * 0.5f;

        float expCloudType = pow(cloudType, (float)properties.renderSettings.cloudTypeExponent);
        // 0.0, 1.0 are constants in shader, but set to aforementioned values.
        // float cloudTypeRemapped = Remap(expCloudType, 0.0f, 1.0f, 0.0f, 1.0f);
            // (float)properties.uniforms.cloudRangeMin, (float)properties.uniforms.cloudRangeMax,
            // (float)properties.uniforms.cloudTypeMin, (float)properties.uniforms.cloudTypeMax);
        return glm::vec4(expCloudType, perlinSample.y, perlinSample.z, perlinSample.w);
    }

    const float PI = 3.141592653589793f;

    // cloud types are remapped from [0,1] so that all values above this become 1
    float CLOUD_COVER_MAX = .8f;

    // fraction of the cloud layer thickness by which the thickness is locally varying at high frequency
    float CLOUD_HEIGHT_VARIATION = .1f;

    // high frequency noises begin to fade at this distance
    float HF_FADE_DISTANCE = 10000.0f;
    // high frequency noises have faded to .5 at this distance
    float HF_END_DISTANCE = 100000.0f;

    // low frequency noises begin to fade at this distance
    float LF_FADE_DISTANCE = 500000.0f;
    // low frequency noises have faded to .5 at this distance
    float LF_END_DISTANCE = 2000000.0f;

    glm::vec4 GetVerticalProfile(glm::vec3 position, CloudProperties &properties) {
        glm::vec2 lngLat = GetSphericalCoords(position);
        glm::vec2 texCoords = glm::vec2(lngLat.x / (2.0f * PI) + 0.5f, 1.0f - lngLat.y / PI + 0.5f);
        // vstr::debug() << "GetVerticalProfile()::pos_over_earth = " << (glm::length(position) - properties.planetRadius) << ", texCoords = " << glm::to_string(texCoords) << std::endl;
        // uCloudTexture = earth-clouds.jpg (black and white)
        // In shader: textureLod(..., 2) call with LOD level 2
        float cloudCoverDensity = GetTexture(properties.cloud.data(), properties.cloudDim, texCoords).r;
        // vstr::debug() << "GetVerticalProfile()::cloudCoverDensity(tex2d) = " << cloudCoverDensity << std::endl;
        float density = Remap(cloudCoverDensity, 0.0f, CLOUD_COVER_MAX, 0.0f, 1.0f);
        glm::vec4 hcomp_with_noise = GetLocalCloudType(texCoords, properties);
        // vstr::debug() << "GetVerticalProfile::local_cloud_type = " << glm::to_string(hcomp_with_noise) << std::endl;

        float cloudType = hcomp_with_noise.r;
        glm::vec3 noiseSample(hcomp_with_noise.g, hcomp_with_noise.b, hcomp_with_noise.a);
        float endHeight = CUMULONIMBUS_END_HEIGHT * (1.0f - CLOUD_HEIGHT_VARIATION * noiseSample.g);
        float startAltitude = properties.planetRadius + CUMULONIMBUS_START_HEIGHT;
        float topAltitude = properties.planetRadius + endHeight;
        float thickness = endHeight - CUMULONIMBUS_START_HEIGHT;

        // "progress" in cloud from bottom to top in range 0 to 1
        float posHeight = glm::length(position);
        // vstr::debug() << "GetVerticalProfile()::Cloud height remap: height = " << posHeight / 1000.0f << ", x = " << startAltitude / 1000.0f << ", y = " << topAltitude / 1000.0f << std::endl;
        float height_in_cloud = Remap(posHeight, startAltitude, topAltitude, 0.0f, 1.0f);

        // vstr::debug() << "remapping " << length(position) / 1000.0f << "km from ["
        //     << (properties.planetRadius + CUMULONIMBUS_START_HEIGHT) / 1000.0f << ", " << topAltitude / 1000.0f << "] to [0, 1]" << std::endl;
        // vstr::debug() << "Sampling at height " << abs(length(position) - properties.planetRadius) / 1000.0f
        //     << "km above planet. Height factor = " << height_in_cloud << std::endl;
        glm::vec2 cloudTypeTexCoords = glm::vec2(cloudType, 1.0f - height_in_cloud);
        // vstr::debug() << "Cloud type tex coords" << glm::to_string(cloudTypeTexCoords) << std::endl;
        glm::vec4 cloudConfig = GetTexture(properties.cloudType.data(), properties.cloudTypeDim, cloudTypeTexCoords, 4, IndexingMode::Clamp);
        // vstr::debug() << "GetVerticalProfile()::cloudConfig = " << glm::to_string(cloudConfig) << std::endl;
        cloudConfig.r *= density * .95f; // glm::vec4(cloudConfig.r * density * .95f, cloudConfig.g, cloudConfig.b, cloudConfig.a);
        // vstr::debug() << "return GetVerticalProfile()::cloudConfig = " << glm::to_string(cloudConfig) << std::endl;
        return cloudConfig;
    }

    glm::vec2 GetCumuloNimbusDensity(glm::vec3 position, CloudProperties &properties) {
        // vstr::debug() << "GetCumuloNimbusDensity()::height = " << glm::length(position) / 1000.0f << std::endl;
        glm::vec4 cloudConfig = GetVerticalProfile(position, properties);
        // vstr::debug() << "Cloud config (vert. profile) = " << glm::to_string(cloudConfig) << std::endl;
        float cloudBase = cloudConfig.r;
        // = 0 => cloudDensity = 0
        // Also, if cloudConfig.a = 0 => total = 0.
        // GetVerticalProfile always returns a vec4 with x, z or y, w components zero, hence
        // either one will always be set to zero.

        float erosionStrength = cloudConfig.g;
        float hfStrength = cloudConfig.b;
        // noiseTexture2D accessed in spherical coordinates
        // getLngLat = spherical coords
        glm::vec4 noise2Dl = GetTexture(properties.noise2d, properties.noise2dDim, GetSphericalCoords(position) * 1.0f);
        // vstr::debug() << "2D noise = " << glm::to_string(noise2Dl) << std::endl;
        glm::vec4 noise2D = GetTexture(properties.noise2d, properties.noise2dDim, GetSphericalCoords(position) * 5.0f);

        float cloudDensity = (float)pow(cloudBase, properties.renderSettings.cloudCoverageExponent);
        // vstr::debug() << "CloudDensity#1: " << cloudDensity << " = pow(cloudDensity[" << cloudDensity << "], "
            // << properties.renderSettings.cloudCoverageExponent << ")" << std::endl;

        float lfInfluence = cloudConfig.g;
        float hfInfluence = hfStrength;
        // if(cameraDist < LF_END_DISTANCE){
        glm::vec3 lf_noise_texCoords(position * (1.0f / properties.renderSettings.cloudLFRepetitionScale));
        // vstr::debug() << "LF Noise3D at pos " << glm::to_string(lf_noise_texCoords) << " = " << glm::to_string(position) << " * " << properties.renderSettings.cloudLFRepetitionScale << "^-1" << std::endl;
        glm::vec4 lfNoises = GetTexture3D(properties.noise, properties.noiseDim, lf_noise_texCoords);
        // vstr::debug() << " = " << glm::to_string(lfNoises) << std::endl;

        // blend between worley and perlin noises using a noise at a different frequency to reduce repetition
        float lr_worley_noise = (1.0f - lfNoises.b) * .8f + lfNoises.r * .2f;
        float lr_whispy_noise = lfNoises.r * .2f + lfNoises.g * .8f;
        float blended_lf_noise = Mix(lr_worley_noise, lr_whispy_noise, noise2Dl.r);
        // when camDist is in the fade out range, the noise is mixed with 0.5
        // blended_lf_noise = mix(blended_lf_noise, .5, remap(cameraDist, LF_FADE_DISTANCE, LF_END_DISTANCE, 0, 1)) * .5 + .5 * noise2D.r;
        blended_lf_noise = .5f * .5f + .5f * noise2D.r;
        // using the formula from Andrew Schneider's SIGGRAPH presentations on Nubis
        float new_cloudDensity = std::clamp(lfInfluence * blended_lf_noise - (lfInfluence - cloudDensity), 0.0f, 1.0f); // clamp(x, 0, 1) = saturate(x) (slide 34/207)
        // vstr::debug() << "CloudDensity#2: " << new_cloudDensity << " =  std::clamp(lfInfluence[" << lfInfluence << "] * blended_lf_noise["
        //     << blended_lf_noise << "] - lfInfluence[" << lfInfluence << "] + cloudDensity[" << cloudDensity << "], 0.0f, 1.0f)" << std::endl;
        cloudDensity = new_cloudDensity;
        // if(high_res && cameraDist < HF_END_DISTANCE){

        glm::vec3 hf_noise_texCoords(position * (1.0f / properties.renderSettings.cloudHFRepetitionScale));
        // vstr::debug() << "HF Noise3D at pos " << glm::to_string(hf_noise_texCoords) << " = " << glm::to_string(position) << " * " << properties.renderSettings.cloudHFRepetitionScale << "^-1" << std::endl;
        glm::vec4 hf_noises = GetTexture3D(properties.noise, properties.noiseDim, hf_noise_texCoords);
        // vstr::debug() << " = " << glm::to_string(hf_noises) << std::endl;

        float hr_worley_noise = (1.0f - hf_noises.b) * .5f + lfNoises.r * .5f;
        float hr_whispy_noise = hf_noises.b * .3f + lfNoises.g * .7f;
        float blended_hf_noise = Mix(hr_worley_noise, hr_whispy_noise, lfNoises.r);
        blended_hf_noise = Mix(blended_hf_noise, .5f,  1.0f); //Remap(cameraDist, HF_FADE_DISTANCE, HF_END_DISTANCE, 0, 1));
        new_cloudDensity = std::clamp(hfInfluence * blended_hf_noise - (hfInfluence - cloudDensity), 0.0f, 1.0f);
        // vstr::debug() << "CloudDensity#2: " << new_cloudDensity << " =  std::clamp(hfInfluence[" << hfInfluence << "] * blended_hf_noise["
        //     << blended_hf_noise << "] - hfInfluence[" << hfInfluence << "] + cloudDensity[" << cloudDensity << "], 0.0f, 1.0f)" << std::endl;
        cloudDensity = new_cloudDensity;
        // vstr::debug() << "CloudDensity#3 = " << cloudDensity << std::endl;

        // }else{
        // Old impl:
        // cloudDensity = std::clamp(hfInfluence * .5f - (hfInfluence - cloudDensity), 0.0f, 1.0f);
        // }
        // }else{
        // // reduce density by assuming noise=0.5 
        // // without this operation, the cloud would become more dense at the LOD region border
        // // MUST BE PERFORMED FOR ALL FUTURE NOISE SCALES TO AVOID DISCONTINUITIES 
        // cloudDensity = std::clamp(lfInfluence * .5f - (lfInfluence - cloudDensity), 0.0f, 1.0f);
        // cloudDensity = std::clamp(hfInfluence * .5f - (hfInfluence - cloudDensity), 0.0f, 1.0f);
        // }
        if (isnan(cloudDensity)) {
            cloudDensity = 0;
        }

        float h = abs(glm::length(position) - properties.planetRadius);
        float height_factor = exp(-h / 8000);

        // What is this actually? Also a density measure?
        float totalDensity = cloudConfig.a * height_factor;
        // if (total > 0.0f) {
        // vstr::debug() << "VerticalProfile(" << glm::to_string(position) << ").a = " << cloudConfig.a << ", height factor = e^(-" << h << "/8000) = "
        //     << height_factor << ". Total = " << total << ", cloud density = " << cloudDensity << " > "
        //     << properties.renderSettings.cloudCutoff << std::endl;
        // }
        // vstr::debug() << "totalDensity = " << totalDensity << ", cloudDensity = " << cloudDensity << std::endl;
        // uCloudCutoff determines the minimum density of a cloud. If < cutoff, density is set to zero,
        // hence uCloudCutoff sets the boundaries of the clouds.
        // cloudConfig.a = cloudBase (The higher the cloud, the thinner it becomes).
        // return glm::vec2(total, cloudDensity);
        return glm::vec2(cloudDensity > properties.renderSettings.cloudCutoff ? totalDensity : 0, cloudDensity);
    }

    glm::vec2 GetCloudDensity(glm::vec3 position, CloudProperties &properties) {
        glm::vec2 acc(0.0f);
        float height = glm::length(position) - properties.planetRadius;
        // vstr::debug() << "Calculating density at " << glm::to_string(position) << std::endl;
        // vstr::debug() << "Height above ground = " << height / 1000.0f << "m, dist to cloud layer = " << std::min(height - CUMULONIMBUS_START_HEIGHT, height - CUMULONIMBUS_END_HEIGHT) / 1000.0f << "m" << std::endl;
        if(height > CUMULONIMBUS_START_HEIGHT && height < CUMULONIMBUS_END_HEIGHT) {
            // vstr::debug() << "GetCloudDensity()::Calculating density at height " << glm::length(position) / 1000.0f << std::endl;
            acc += GetCumuloNimbusDensity(position, properties);
            // vstr::debug() << "Density = " << glm::to_string(acc) << std::endl;
        }
        return acc;
    }
}

#endif