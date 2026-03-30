#include "Tree.hpp"
#include <cmath>

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
    
    Subdivide(ROOT_NODE_INDEX);
    // Calculate new bounding boxes of 8 child nodes
    // Compute density at (aabbMax - aabbMin) * 0.5 (centre of bounding box) to sample density in the node.
    // If density is very large/ very low/ max depth reached, abort.
}

void Tree::Subdivide(unsigned int index) {
    // usedNodeIndex starts at 0, maxNodeCount starts at 1, so ">="
    if (usedNodeIndex + 8 >= maxNodeCount) // If no space for 8 child nodes anymore => leaf node
        return;
    
    auto &node = nodes[index];
    // Nodes are stored sequentially, so children of current node are the next 8 nodes in the nodes-array.
    node.firstChildIndex = usedNodeIndex + 1;
    for (int i = 0; i < 8; i++) {
        unsigned int currNodeIndex = ++usedNodeIndex;
        UpdateBounds(currNodeIndex, i);
        Subdivide(currNodeIndex);
    }
}

void Tree::UpdateBounds(unsigned int index, unsigned int relChildIndex) {
    // Add break-condition if density is very large/ very low?
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
    //   /               / |
    //  /               /  |
    // ----------------/   |
    // |   |           |   |
    // |   |    x      |   |
    // |   |-----------|---| back
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

    if (usedNodeIndex + 8 >= maxNodeCount) { // Is leaf node? (cannot use node.IsLeaf(), as node.leftChildNode hasnt been set yet)
        // Sample density at the centre of current node.
        // (Or: calculate the average in the corners?)
        node.val = GetDensity(centre);
    }
}

float Tree::GetDensity(glm::vec3 pos) {
    throw;
}