#include "Tree.hpp"
#include <cmath>

Tree::Tree(glm::uvec3 dimensions, unsigned int maxDepth) {
    this->dimensions = dimensions;
    this->usedNodes = 0;

    unsigned int N = (int)pow(2, maxDepth) - 1;
    nodes = std::make_unique()<TreeNode[]>(N);
}

void Tree::Build(std::vector<float> noise, std::vector<float> noise2d) {
    unsigned int currNodeIndex = 0;
    unsigned int currDepth = 0;
    auto &rootNode = nodes[currNodeIndex];
    rootNode.aabbMax = GetIndexFromPos(dimensions); // Set max bounds for root node AABB
    
    currDepth += 1;
    auto &leftChildNode = nodes[++currNodeIndex];
    auto &rightChildNode = nodes[++currNodeIndex];

    leftChildNode.depth = currDepth;
    rightChildNode.depth = currDepth;

    // Calculate new bounding boxes of 8 child nodes
    // Compute density at (aabbMax - aabbMin) * 0.5 (centre of bounding box) to sample density in the node
    // If density is very large/ very low/ max depth reached, abort.
}

unsigned int Tree::GetIndexFromPos(glm::uvec3 pos) const {
    return pos.x * dimensions.x + pos.y * dimensions.y + pos.z * dimensions.z;
}

glm::uvec3 Tree::GetPosFromIndex(unsigned int index) const {
    glm::uvec3 pos;
    pos.x = index % dimensions.x;
    pos.y = int(index / dimensions.x) % dimensions.y;
    pos.z = int(index / (dimensions.x * dimensions.y));
    return pos;
}