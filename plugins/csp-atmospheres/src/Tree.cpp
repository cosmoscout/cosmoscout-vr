////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Tree.hpp"
#include <cmath>

namespace csp::atmospheres {
    Tree::Tree(glm::uvec3 dimensions, unsigned int maxDepth, CloudProperties properties) {
        this->dimensions = dimensions;
        this->maxDepth = maxDepth;
        usedNodeIndex = 0;
        maxNodeCount = (int)pow(8, maxDepth) + 1; // Per level of depth, each node subdivides into 8 child nodes, plus the root node.
        this->properties = std::move(properties); // Correct usage of std::move?
        nodes = std::make_unique<TreeNode[]>(maxNodeCount);
    }

    void Tree::Build() {
        const unsigned int ROOT_NODE_INDEX = 0;
        auto &rootNode = nodes[ROOT_NODE_INDEX];
        rootNode.aabbMax = (glm::vec3)dimensions; // Set dimensions as max bounds for root node.
        
        unsigned int depth = 0;
        usedNodeIndex = 0;
        Subdivide(ROOT_NODE_INDEX, depth);
    }

    void Tree::Subdivide(unsigned int index, unsigned int depth) {
        if (depth >= maxDepth) // If level of depth has been reached, stop subdivision process.
            return;
        if (GetAverageDensity(index) <= 1e-4) // If average density throughout the node is too small, stop subdividing.
            return;

        depth += 1;
        // Decide if node should be subdivided
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
        auto &parent = nodes[index];

        glm::vec3 parentExtends = parent.aabbMax - parent.aabbMin;
        glm::vec3 centre = parentExtends * 0.5f;

        // First, scale the node bounds.
        node.aabbMax = node.aabbMin + centre;

        glm::vec3 boundsTransformation(0.0);
        int horizontalMode = relChildIndex % 4; // Front left, right, back left, right
        int verticalOrder = relChildIndex / 4; // Upper, lower level

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
            boundsTransformation += glm::vec3(centre.x, 0.0, 0.0);
        }
        if (back) {
            boundsTransformation += glm::vec3(0.0, 0.0, centre.z);
        }
        if (top) {
            boundsTransformation += glm::vec3(0.0, centre.y, 0.0);
        }

        node.aabbMin += boundsTransformation;
        node.aabbMax += boundsTransformation;
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

        float densitySum = 0.0f;
        for (size_t i = 0; i < DENSITY_SAMPLE_COUNT; i++) {
            densitySum += GetDensity(densitySampleLocations[i]);
        }

        return densitySum / DENSITY_SAMPLE_COUNT;
    }
}