////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Tree.hpp"
#include <cmath>

namespace csp::atmospheres {
    const unsigned int ROOT_NODE_INDEX = 0;

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
        if (depth >= maxDepth) // If level of depth has been reached, stop subdivision process.
            return;
        
        float avgDensity = GetAverageDensity(index);
        // if (depth == maxDepth - 1)
        //     vstr::debug() << "Average density at index " << index << " = " << avgDensity << std::endl;
        // if (avgDensity <= 1e-3)
        //     return;

        depth += 1;
        auto &node = nodes[index];
        // Nodes are stored sequentially, so children of current node are the next 8 nodes in the nodes-array.
        node.firstChildIndex = usedNodeIndex + 1;
        for (int i = 0; i < 8; i++) {
            unsigned int currNodeIndex = ++usedNodeIndex;
            UpdateBounds(currNodeIndex, i);
            Subdivide(currNodeIndex, depth);
        }
    }

    void Tree::UpdateBounds(unsigned int index, unsigned int relChildIndex) {
        auto &node = nodes[index];
        auto &parent = nodes[index - 1];

        glm::vec3 parentExtends = parent.aabbMax - parent.aabbMin;
        glm::vec3 centre = parentExtends * 0.5f;

        // First, scale the node bounds so that the bounding box has the right size.
        node.aabbMax = parent.aabbMin + centre;

        glm::vec3 boundsTransformation = parent.aabbMin;
        // int horizontalMode = relChildIndex % 4; // Front left, right, back left, right
        // int verticalOrder = relChildIndex / 4; // Upper, lower level

        // According to the relative position inside the parent node, transform the child node along each axis.
        bool right = relChildIndex % 2 == 1; // Rightward nodes
        bool back = relChildIndex == 2 || relChildIndex == 3 || relChildIndex == 6 || relChildIndex == 7; // Backward nodes
        bool top = relChildIndex / 4 >= 1; // Upper nodes

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

        if (right) {
            boundsTransformation.x = centre.x; // += glm::vec3(centre.x, 0.0, 0.0);
        }
        if (back) {
            boundsTransformation.z = centre.z; // += glm::vec3(0.0, 0.0, centre.z);
        }
        if (top) {
            boundsTransformation.y = centre.y; // += glm::vec3(0.0, centre.y, 0.0);
        }

        node.aabbMin = boundsTransformation;
        node.aabbMax += node.aabbMin - boundsTransformation;
        // vstr::debug() << "Node " << index << " (rel = " << relChildIndex << ") new bounds = " << glm::to_string(node.aabbMin) << ", " << glm::to_string(node.aabbMax) << std::endl;
    }

    float Tree::GetDensity(glm::vec3 pos) {
        glm::vec2 density = GetCloudDensity(pos, properties);
        return density.x; // First component: with cutoff (actual visible density)
    }

    float Tree::GetAverageDensity(unsigned int index) {
        auto &node = nodes[index];

        const unsigned int DENSITY_SAMPLE_COUNT = 9;
        float densitySamples[DENSITY_SAMPLE_COUNT] = {};
        glm::vec3 densitySampleLocations[DENSITY_SAMPLE_COUNT];
        // Sample density at node corners and the centre.
        // Subdivide if average density is above cutoff.
        densitySampleLocations[0] = node.aabbMin;
        densitySampleLocations[1] = node.aabbMax;
        densitySampleLocations[2] = glm::vec3(node.aabbMin.x, node.aabbMin.x, node.aabbMax.z); // LL
        densitySampleLocations[3] = glm::vec3(node.aabbMin.x, node.aabbMax.y, node.aabbMax.z); // UL
        densitySampleLocations[4] = glm::vec3(node.aabbMax.x, node.aabbMin.y, node.aabbMax.z); // LR
        densitySampleLocations[5] = glm::vec3(node.aabbMax.x, node.aabbMax.y, node.aabbMin.z); // UR
        densitySampleLocations[6] = glm::vec3(node.aabbMax.x, node.aabbMax.y, node.aabbMin.z); // UM
        densitySampleLocations[7] = glm::vec3(node.aabbMax.x, node.aabbMin.y, node.aabbMax.z); // LM
        densitySampleLocations[8] = node.aabbMax - node.aabbMin; // Centre

        // vstr::debug() << "Densities in node " << index << ": ";
        // for (size_t i = 0; i < DENSITY_SAMPLE_COUNT; i++) {
        //     float density = GetDensity(densitySampleLocations[i]);
        //     vstr::debug() << density << ", ";
        // }
        // vstr::debug() << std::endl;

        float densitySum = 0.0f;
        for (size_t i = 0; i < DENSITY_SAMPLE_COUNT; i++) {
            float density = GetDensity(densitySampleLocations[i]);
            densitySum += density;
        }

        return densitySum / DENSITY_SAMPLE_COUNT;
    }
}