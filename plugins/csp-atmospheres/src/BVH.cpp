////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "BVH.hpp"
#include <memory>
#include <vector>

BVH::BVH(unsigned int objCount) {
    size = 2 * objCount - 1;
    this->objCount = objCount;
    nodeCount = 0;

    nodes = std::make_unique<BVHNode[]>(size);
    objIndices = std::make_unique<unsigned int[]>(objCount);
}

void BVH::UpdateNodeBounds(unsigned int nodeIndex) {

}

void BVH::Subdivide(unsigned int nodeIndex) {

}

void BVH::Build() {
    
}