////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_TREE_BVH_HPP
#define CSP_TREE_BVH_HPP

#include <glm/glm.hpp>
#include <memory>
#include <vector>

struct TreeNode {
    // Index-based retrieval (3D noise texture stored as vector<float>)
    glm::vec3 aabbMin, aabbMax;
    // rightChild != 0 => Leaf node, so leftChild = first
    unsigned int children[8];

    TreeNode() {
        aabbMin = glm::vec3(0.0);
        aabbMax = glm::vec3(0.0);
        for (int i = 0; i < 8; i++) {
            children[i] = 0;
        }
    }
};

class Tree {
private:
    glm::uvec3 dimensions;
    unsigned int usedNodes;
    std::unique_ptr<TreeNode[]> nodes;

    // Calculates cloud density at the given index
    static float GetCost(std::vector<float> noise, std::vector<float> noise2d, glm::vec3 pos) {
        return 1.0;
    }

public:
    Tree(glm::uvec3 dimensions, unsigned int maxDepth);
    void Build(std::vector<float> noise, std::vector<float> noise2d);

    unsigned int GetIndexFromPos(glm::uvec3 pos) const;
    glm::uvec3 GetPosFromIndex(unsigned int index) const;
};

#endif