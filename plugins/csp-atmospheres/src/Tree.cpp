#include "Tree.hpp"

Tree::Tree(glm::uvec3 dimensions, unsigned int maxDepth) {
    this->dimensions = dimensions;
    this->usedNodes = 0;
    costData = std::vector<float>();

    unsigned int N = (int)pow(2, maxDepth) - 1;
    nodes = std::make_unique<TreeNode[]>(N);
}

void Tree::GenerateCostData(std::vector<float> noise, std::vector<float> noise2d) {
    for (size_t i = 0; i < dimensions.x; i++) {
        for (size_t j = 0; j < dimensions.y; j++) {
            for (size_t k = 0; k < dimensions.z; k++) {
                glm::uvec3 pos(i, j, k);
                unsigned int index = GetIndexFromPos(pos);
                float cost = GetCost(noise, noise2d, index);
                costData[index] = cost;
            }
        }
    }
}

void Tree::Build() {
    unsigned int currNodeIndex = 0;
    unsigned int currDepth = 0;
    auto &rootNode = nodes[currNodeIndex];
    rootNode.aabbMax = GetIndexFromPos(dimensions); // Set max bounds for root node AABB
    
    currDepth += 1;
    auto &leftChildNode = nodes[++currNodeIndex];
    auto &rightChildNode = nodes[++currNodeIndex];

    leftChildNode.depth = currDepth;
    rightChildNode.depth = currDepth;

    // Calculate split position between nodes
    // Compute new bounding boxes
    // TODO
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