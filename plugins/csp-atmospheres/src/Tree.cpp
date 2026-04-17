////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Tree.hpp"
#include <cmath>

namespace csp::atmospheres {
    const unsigned int ROOT_NODE_INDEX = 0;
    const unsigned int BASE_DENSITY_SAMPLES = 10000;

    Tree::Tree(glm::vec3 totalBoundsMin, glm::vec3 totalBoundsMax, unsigned int maxDepth, CloudProperties properties) {
        this->maxDepth = maxDepth;
        usedNodeIndex = 0;

        maxNodeCount = 0;
        for (size_t i = 0; i <= maxDepth; i++) {
            maxNodeCount += (int)pow(8, i);
        }
        vstr::debug() << "Creating octree with max size " << maxNodeCount << std::endl;

        this->properties = std::move(properties); // Correct usage of std::move?
        nodes = std::make_unique<TreeNode[]>(maxNodeCount);
        
        // Configure total bounds of the root node.
        auto &rootNode = nodes[ROOT_NODE_INDEX];
        rootNode.aabbMin = totalBoundsMin;
        rootNode.aabbMax = totalBoundsMax;
    }

    void Tree::Build() {        
        unsigned int depth = 0;
        usedNodeIndex = 0;
        Subdivide(ROOT_NODE_INDEX, depth);
    }

    void Tree::Subdivide(unsigned int index, unsigned int depth) {
        // vstr::debug() << "Depth " << depth << "(i = " << index << "): aabb = "
        //     << glm::to_string(nodes[index].aabbMin * 0.00001f) << ", "
        //     << glm::to_string(nodes[index].aabbMax * 0.00001f) << std::endl;
        // for (size_t i = 0; i < depth; i++)
        // {
        //     vstr::debug() << " ";
        // }
        // vstr::debug() << "[Depth = " << depth << "] index " << index << ": ";

        if (depth >= maxDepth) { // If level of depth has been reached, stop subdivision process.
            // vstr::debug() << "max depth reached. STOP" << std::endl;
            return;
        }
        
        // float depthFactor = 1.0f / (depth + 1);
        // unsigned int sampleCount = std::max(static_cast<unsigned int>(((float)BASE_DENSITY_SAMPLES) * depthFactor));
        // vstr::debug() << "Running " << sampleCount << " samples on depth " << depth << std::endl;
        float totalDensity = GetTotalDensity(index, BASE_DENSITY_SAMPLES * (depth + 1));
        if (totalDensity <= 1e-3) {
            // vstr::debug() << "zero density. STOP" << std::endl;
            return;
        }
        //else {
            // for (size_t i = 0; i < depth; i++)
            // {
            //     vstr::debug() << " ";
            // }
            // vstr::debug() << "[Depth = " << depth << "] index " << index << " = " << totalDensity << std::endl;
            // vstr::debug() << " = " << totalDensity << std::endl;
        //}

        depth += 1;
        auto &node = nodes[index];
        // Nodes are stored sequentially, so children of current node are the next 8 nodes in the nodes-array.
        node.firstChildIndex = usedNodeIndex + 1;

        // Update children node bounds
        //           top
        //    -----------------m
        //   /|              / |
        //  / |             /  |
        // ----------------/   |
        // |  |            |   |
        // |  |     x      |   |
        // |  |------------|---| back
        // |  /            |  /
        // | /             | /
        // n---------------|/ right
        //
        // n = aabbMin
        // m = aabbMax
        // x = centre of node

        glm::vec3 centre = (node.aabbMax - node.aabbMin) * 0.5f;
        if (glm::length(centre) > 1e-4)
            centre = node.aabbMin + centre;
        glm::vec3 newMinBounds[8], newMaxBounds[8];
        // Bottom
        newMinBounds[0] = node.aabbMin;
        newMinBounds[1] = glm::vec3(centre.x, node.aabbMin.y, node.aabbMin.z);
        newMinBounds[2] = glm::vec3(node.aabbMin.x, centre.y, node.aabbMin.z);
        newMinBounds[3] = glm::vec3(centre.x, centre.y, node.aabbMin.z);
        // Top
        newMinBounds[4] = glm::vec3(node.aabbMin.x, node.aabbMin.y, centre.z);
        newMinBounds[5] = glm::vec3(centre.x, node.aabbMin.y, centre.z);
        newMinBounds[6] = glm::vec3(node.aabbMin.x, centre.y, centre.z);
        newMinBounds[7] = centre;

        // Bottom
        newMaxBounds[0] = centre;
        newMaxBounds[1] = glm::vec3(node.aabbMax.x, centre.y, centre.z);
        newMaxBounds[2] = glm::vec3(centre.x, node.aabbMax.y, centre.z);
        newMaxBounds[3] = glm::vec3(node.aabbMax.x, node.aabbMax.y, centre.z);
        // Top
        newMaxBounds[4] = glm::vec3(centre.x, centre.y, node.aabbMax.z);
        newMaxBounds[5] = glm::vec3(node.aabbMax.x, centre.y, node.aabbMax.z);
        newMaxBounds[6] = glm::vec3(centre.x, node.aabbMax.y, node.aabbMax.z);
        newMaxBounds[7] = node.aabbMax;

        // vstr::debug() << "Node bounds = " << glm::to_string(node.aabbMin) << ", " << glm::to_string(node.aabbMax) << std::endl;

        for (int i = 0; i < 8; i++) {
            unsigned int currNodeIndex = ++usedNodeIndex;
            
            auto &childNode = nodes[currNodeIndex];
            childNode.aabbMin = newMinBounds[i];
            childNode.aabbMax = newMaxBounds[i];

            Subdivide(currNodeIndex, depth);
            // if (depth == maxDepth)
            //     vstr::debug() << "Node " << currNodeIndex << " bounds = " << glm::to_string(childNode.aabbMin) << ", " << glm::to_string(childNode.aabbMax) << std::endl;
        }
    }

    float Tree::GetDensity(glm::vec3 pos) {
        glm::vec2 density = GetCloudDensity(pos, properties);
        return density.x;
    }

    float Tree::GetTotalDensity(unsigned int index, unsigned int totalSamples = BASE_DENSITY_SAMPLES) {
        auto &node = nodes[index];

        // Sample at random positions inside the bounding box
        float totalDensity = 0.0f;
        for (size_t i = 0; i < totalSamples; i++) {
            glm::vec3 randomUnitPos = glm::vec3(rand(), rand(), rand()) * (1.0f / RAND_MAX);
            glm::vec3 randomSamplePos = node.aabbMin + node.GetExtends() * randomUnitPos;
            totalDensity += GetDensity(randomSamplePos);
        }
        return totalDensity;
    }

    // float Tree::GetTotalDensity(unsigned int index) {
    //     auto &node = nodes[index];

    //     const unsigned int DENSITY_SAMPLES = 500;
    //     glm::vec3 mainExtends = node.aabbMax - node.aabbMin;

    //     glm::vec3 upperLeftBack = glm::vec3(node.aabbMin.x, node.aabbMax.y, node.aabbMax.z); // UL
    //     glm::vec3 lowerRightFront = glm::vec3(node.aabbMax.x, node.aabbMin.y, node.aabbMin.z); // LR
    //     glm::vec3 mainCrossExtends = upperLeftBack - lowerRightFront;

    //     glm::vec3 upperRightFront = glm::vec3(node.aabbMax.x, node.aabbMin.y, node.aabbMax.z);
    //     glm::vec3 lowerLeftBack = glm::vec3(node.aabbMin.x, node.aabbMax.y, node.aabbMin.z);
    //     glm::vec altExtends = upperRightFront - lowerLeftBack;

    //     glm::vec3 upperLeftFront = glm::vec3(node.aabbMin.x, node.aabbMax.y, node.aabbMax.z);
    //     glm::vec3 lowerRightBack = glm::vec3(node.aabbMax.x, node.aabbMax.y, node.aabbMin.z);
    //     glm::vec3 altCrossExtends = upperLeftFront - lowerRightBack;

    //     // TODO: as the octree splits 8 times along the same extends every subdivision iteration,
    //     // the average density along the same 4 diagonals will be calculated again and again.
    //     // Instead, sample along random points inside the cubes? Or along lines parallel to the coordinate axes?

    //     float totalDensity = 0.0;
    //     unsigned int totalSamples = 0;
    //     for (size_t i = 1; i < DENSITY_SAMPLES; i++) {
    //         float i_f = (float)i;
    //         float coeff = float(i_f / DENSITY_SAMPLES);
    //         glm::vec3 mainSamplePos = node.aabbMin + coeff * mainExtends;
    //         glm::vec3 mainCrossSamplePos = upperLeftBack + coeff * mainCrossExtends;
    //         glm::vec3 altSamplePos = lowerRightBack + coeff * altExtends;
    //         glm::vec3 altCrossSamplePos = upperLeftFront + coeff * altCrossExtends;
    //         totalDensity += GetDensity(mainSamplePos) + GetDensity(mainCrossSamplePos) + GetDensity(altSamplePos) + GetDensity(altCrossSamplePos);
    //         totalSamples += 4;
    //     }

    //     return totalDensity; // / totalSamples;
    // }
}