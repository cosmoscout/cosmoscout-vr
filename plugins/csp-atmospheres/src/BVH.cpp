////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "BVH.hpp"
#include <memory>
#include <vector>

BVH::BVH(unsigned int objCount, std::vector<BVHObject> &objs) {
    nodesSize = 2 * objCount - 1;
    this->objCount = objCount;
    nodesUsed = 0;

    nodes = std::make_unique<BVHNode[]>(nodesSize);
    objIndices = std::make_unique<unsigned int[]>(objCount);
    for (unsigned int i = 0; i < objCount; i++) {
        objIndices[i] = i;
    }
    this->objs = objs;
}

void BVH::UpdateNodeBounds(unsigned int nodeIndex) {
    auto &node = nodes[nodeIndex];
    node.UpdateBounds(objs);
}

void BVH::Subdivide(unsigned int nodeIndex) {
    auto &node = nodes[nodeIndex];
    if (node.objCount <= 2) // If node manages <= 2 objects, stop the subdivision process.
        return;
    
    auto extend = node.aabbMax - node.aabbMin;
    int axis = 0;
    if (extend.y > extend.x)
        axis = 1;
    if (extend.z > extend[axis])
        axis = 2;

    // Split the bounding box along its longest extend
    auto splitPos = node.aabbMin[axis] + extend[axis] * 0.5f;
    // Case 1: only branch nodes so far, hence
    // leftFirst = first index of object managed by this node, objCount = number of objects managed
    unsigned int i = node.leftFirst;
    unsigned int j = i + node.objCount - 1;
    while (i <= j) {
        if (objs[objIndices[i]].GetCentre()[axis] < splitPos)
            i++;
        else // Object is right of axis split
            std::swap(objIndices[i], objIndices[j--]);
    }
    // i increment stops at the index of object that was moved to the left node
    unsigned int leftFirst = i - node.leftFirst;
    // Abort if left node or right node are empty
    if (leftFirst == 0 || leftFirst == node.objCount)
        return;
    
    unsigned int leftChildIndex = nodesUsed++;
    unsigned int rightChildIndex = nodesUsed++;
    // Create new child nodes by splitting node into left and right part
    nodes[leftChildIndex].leftFirst = node.leftFirst;
    nodes[leftChildIndex].objCount = leftFirst;
    nodes[rightChildIndex].leftFirst = i;
    nodes[rightChildIndex].objCount = node.objCount - leftFirst;

    node.leftFirst = leftChildIndex;
    node.objCount = 0; // Convert leaf node to branch node
    UpdateNodeBounds(leftChildIndex);
    UpdateNodeBounds(rightChildIndex);
    Subdivide(leftChildIndex);
    Subdivide(rightChildIndex);
}

const size_t BVH_ROOT_NODE_INDEX = 0;

void BVH::Build() {
    auto rootNode = nodes[BVH_ROOT_NODE_INDEX];
    rootNode.leftFirst = 0;
    rootNode.objCount = objCount;
    nodesUsed += 1;
    UpdateNodeBounds(BVH_ROOT_NODE_INDEX);
    Subdivide(BVH_ROOT_NODE_INDEX);
}