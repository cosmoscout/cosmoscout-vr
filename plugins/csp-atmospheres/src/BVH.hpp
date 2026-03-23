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
#include <optional>
#include <VistaMath/VistaBoundingBox.h>

#include <glm/glm.hpp>

struct BVHObject {
private:
    glm::vec3 centre; // Central pos
    std::optional<std::pair<glm::vec3, glm::vec3>> aabb;
public:
    std::vector<glm::vec3> verts;

    BVHObject(std::vector<glm::vec3> verts) {
        // Automatically find central pos of object = (v_1 + ... + v_N)) / N. (?)
        size_t coeff = 1 / verts.size();
        for (auto &v : verts) {
            auto side = v;
            v *= coeff;
            centre += v;
        }
        this->verts = verts;
        this->aabb = {};
    }

    BVHObject(std::vector<glm::vec3> verts, glm::vec3 aabbMin, glm::vec3 aabbMax) : BVHObject(verts) {
        this->aabb = std::make_pair(aabbMin, aabbMax);
    }

    glm::vec3 GetCentre() const {
        return centre;
    }

    std::optional<std::pair<glm::vec3, glm::vec3>> GetAABB() const {
        return aabb;
    }
};

struct BVHNode {
    glm::vec3 aabbMin, aabbMax; // Bounding box that partitions the scene
    // (Branch node) If objCount == 0, leftFirst is the index of the left child node
    // (Leaf node)   If objCount > 0, it is the index of the first object stored by this node
    unsigned int leftFirst;
    unsigned int objCount;

    bool IsLeaf() const {
        return objCount > 0;
    }

    BVHNode() {
        aabbMin = glm::vec3(1e10);
        aabbMax = glm::vec3(-1e10);
        leftFirst = objCount = 0;
    }

    void UpdateBounds(std::vector<BVHObject> &objs) {
        for (auto &obj : objs) {
            for (auto &v : obj.verts) {
                aabbMin = glm::min(aabbMin, v);
                // aabbMin.x = fminf(aabbMin.x, v.x);
                // aabbMin.y = fminf(aabbMin.y, v.y);
                // aabbMin.z = fminf(aabbMin.z, v.z);

                aabbMax = glm::max(aabbMax, v);
                // aabbMax.x = fmaxf(aabbMax.x, v.x);
                // aabbMax.y = fmaxf(aabbMax.y, v.y);
                // aabbMax.z = fmaxf(aabbMax.z, v.z);
            }
        }
    }
};

class BVH {
private:
    unsigned int nodesSize, objCount, nodesUsed;
    std::unique_ptr<BVHNode[]> nodes;
    std::vector<BVHObject> objs;
    std::unique_ptr<unsigned int[]> objIndices;

    void UpdateNodeBounds(unsigned int nodeIndex);
    void Subdivide(unsigned int nodeIndex);

public:
    BVH(unsigned int objCount, std::vector<BVHObject> &objs);
    void Build();
};

#endif