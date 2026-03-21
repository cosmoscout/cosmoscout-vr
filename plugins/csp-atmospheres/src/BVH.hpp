////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_ATMOSPHERES_BVH_HPP
#define CSP_ATMOSPHERES_BVH_HPP

// Total size: 32 bytes
// (Optimised for 64 byte GPU cache lines, thus two nodes fit into one line)
#include <memory>
#include <vector>
typedef struct BVHNode {
    VistaBoundingBox aabb; // Bounding box that partitions the scene
    // (Branch node) If objCount == 0, leftFirst is the index of the left child node
    // (Leaf node)   If objCount > 0, it is the index of the first object stored by this node
    unsigned int leftFirst;
    unsigned int objCount;

    bool IsLeaf() const {
        return objCount > 0;
    }
};

class BVH {
private:
    unsigned int size, objCount, nodeCount;
    std::unique_ptr<BVHNode[]> nodes;
    std::unique_ptr<unsigned int[]> objIndices;

    void UpdateNodeBounds(unsigned int nodeIndex);
    void Subdivide(unsigned int nodeIndex);

public:
    BVH(unsigned int objCount);

    void Build();
};

#endif