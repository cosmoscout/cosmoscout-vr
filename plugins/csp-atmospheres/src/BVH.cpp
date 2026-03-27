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
// ---------------------
// --- BVH Generator ---
// ---------------------

int BVHGenerator::GetIndexFromPos(glm::ivec3 pos) {
    return pos.x * dimensions.x + pos.y * dimensions.y + pos.z;
}

glm::vec3 BVHGenerator::GetNoise(glm::ivec3 pos) {
    int index = GetIndexFromPos(pos);
    glm::vec3 noise(0.0f);
    noise.x = noiseTexture[index];
    noise.y = noiseTexture[index + 1];
    noise.z = noiseTexture[index + 2];
    return noise;
}

float BVHGenerator::GetDensity(glm::ivec3 pos) {
    // glm::vec3 val = GetNoise(pos);
    // float lr_worley_noise = (1 - val.b) * .8 + val.r * .2;
    // float lr_whispy_noise = val.r * .2 + val.g * .8;
    // using the formula from Andrew Schneider's SIGGRAPH presentations on Nubis

    return 1.0; // TODO
}

BVHObject BVHGenerator::GenerateBVHObject(glm::ivec3 pos) {
    float density = GetDensity(pos);
    if (density > densityCutoff) { // This cell's density is tangible enough to be seen and taken into account for the raymarch.
    }
    throw;
}

std::vector<BVHObject> BVHGenerator::GenerateBVHObjectsFromNoise() {
    throw;
}

BVHGenerator::BVHGenerator(glm::ivec3 dimensions, float *noiseTexture, float *noise2DTexture,
    float coverageExp, float densityCoeff, float densityCutoff) {
    this->dimensions = dimensions;
    this->noiseTexture = noiseTexture;
    this->coverageExp = coverageExp;
    this->densityCoeff = densityCoeff;
    this->densityCutoff = densityCutoff;
}

BVH GenerateBVH() {
    throw;
}