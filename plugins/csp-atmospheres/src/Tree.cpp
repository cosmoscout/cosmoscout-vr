////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Tree.hpp"
#include <cmath>

namespace csp::atmospheres {
    const unsigned int ROOT_NODE_INDEX = 0;
    const unsigned int BASE_DENSITY_SAMPLES = 10000;

    Tree::Tree(glm::vec3 totalBoundsMin, glm::vec3 totalBoundsMax, unsigned int maxDepth, CloudProperties properties, bool debug) {
        this->maxDepth = maxDepth;
        usedNodeIndex = 0;

        maxNodeCount = 0;
        for (size_t i = 0; i <= maxDepth; i++) {
            maxNodeCount += (int)pow(8, i);
        }
        vstr::debug() << "Creating octree with max size " << maxNodeCount << std::endl;

        this->properties = std::move(properties); // Correct usage of std::move?
        nodes = std::make_unique<TreeNode[]>(maxNodeCount);
        
        // Configure total bounds of the root node.
        auto &rootNode = nodes[ROOT_NODE_INDEX];
        rootNode.aabbMin = totalBoundsMin;
        rootNode.aabbMax = totalBoundsMax;

        debugMode = debug;
        debugShader = VistaGLSLShader();
        vbo = std::make_unique<VistaBufferObject>();
        ibo = std::make_unique<VistaBufferObject>();
        vao = std::make_unique<VistaVertexArrayObject>();
    }

    void Tree::Build() {        
        unsigned int depth = 0;
        usedNodeIndex = 0;
        Subdivide(ROOT_NODE_INDEX, depth);
    }

    unsigned int Tree::Subdivide(unsigned int index, unsigned int depth) {
        // vstr::debug() << "Depth " << depth << "(i = " << index << "): aabb = "
        //     << glm::to_string(nodes[index].aabbMin * 0.00001f) << ", "
        //     << glm::to_string(nodes[index].aabbMax * 0.00001f) << std::endl;
        if (debugMode) {
            for (size_t i = 0; i < depth; i++)
            {
                vstr::debug() << " ";
            }
            vstr::debug() << "[Depth = " << depth << "] index " << index << ": ";
        }

        auto &node = nodes[index];

        float totalDensity = GetTotalDensity(index, BASE_DENSITY_SAMPLES * (depth + 1));
        if (totalDensity <= 1e-3) {
            if (debugMode)
                vstr::debug() << "zero density. STOP" << std::endl;

                node.density = 0;
            return 0;
        }
        else {
            if (debugMode)
                vstr::debug() << "density = " << totalDensity << ". ";

                node.density = totalDensity;
        }

        if (depth >= maxDepth) { // If level of depth has been reached, stop subdivision process.
            if (debugMode)
                vstr::debug() << "max depth reached. STOP" << std::endl;;

            return 0;
        }
        if (debugMode)
            vstr::debug() << std::endl;

        depth += 1;
        // Nodes are stored sequentially, so children of current node are the next 8 nodes in the nodes-array.

        // Update children node bounds
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

        glm::vec3 centre = (node.aabbMax - node.aabbMin) * 0.5f;
        if (glm::length(centre) > 1e-4)
            centre = node.aabbMin + centre;
        glm::vec3 newMinBounds[8], newMaxBounds[8];
        // Bottom
        newMinBounds[0] = node.aabbMin;
        newMinBounds[1] = glm::vec3(centre.x, node.aabbMin.y, node.aabbMin.z);
        newMinBounds[2] = glm::vec3(node.aabbMin.x, centre.y, node.aabbMin.z);
        newMinBounds[3] = glm::vec3(centre.x, centre.y, node.aabbMin.z);
        // Top
        newMinBounds[4] = glm::vec3(node.aabbMin.x, node.aabbMin.y, centre.z);
        newMinBounds[5] = glm::vec3(centre.x, node.aabbMin.y, centre.z);
        newMinBounds[6] = glm::vec3(node.aabbMin.x, centre.y, centre.z);
        newMinBounds[7] = centre;

        // Bottom
        newMaxBounds[0] = centre;
        newMaxBounds[1] = glm::vec3(node.aabbMax.x, centre.y, centre.z);
        newMaxBounds[2] = glm::vec3(centre.x, node.aabbMax.y, centre.z);
        newMaxBounds[3] = glm::vec3(node.aabbMax.x, node.aabbMax.y, centre.z);
        // Top
        newMaxBounds[4] = glm::vec3(centre.x, centre.y, node.aabbMax.z);
        newMaxBounds[5] = glm::vec3(node.aabbMax.x, centre.y, node.aabbMax.z);
        newMaxBounds[6] = glm::vec3(centre.x, node.aabbMax.y, node.aabbMax.z);
        newMaxBounds[7] = node.aabbMax;

        // vstr::debug() << "Node bounds = " << glm::to_string(node.aabbMin) << ", " << glm::to_string(node.aabbMax) << std::endl;
        unsigned int children = 8;
        for (int i = 0; i < 8; i++) {
            unsigned int currNodeIndex = ++usedNodeIndex;
            
            auto &childNode = nodes[currNodeIndex];
            childNode.aabbMin = newMinBounds[i];
            childNode.aabbMax = newMaxBounds[i];

            children += Subdivide(currNodeIndex, depth);
            // if (depth == maxDepth)
            //     vstr::debug() << "Node " << currNodeIndex << " bounds = " << glm::to_string(childNode.aabbMin) << ", " << glm::to_string(childNode.aabbMax) << std::endl;
        }
        node.childrenCount = children;
        return children;
    }

    float Tree::GetDensity(glm::vec3 pos) {
        glm::vec2 density = GetCloudDensity(pos, properties);
        return density.x;
    }

    float Tree::GetTotalDensity(unsigned int index, unsigned int totalSamples = BASE_DENSITY_SAMPLES) {
        auto &node = nodes[index];

        // Sample at random positions inside the bounding box
        float totalDensity = 0.0f;
        for (size_t i = 0; i < totalSamples; i++) {
            glm::vec3 randomUnitPos = glm::vec3(rand(), rand(), rand()) * (1.0f / RAND_MAX);
            glm::vec3 randomSamplePos = node.aabbMin + node.GetExtends() * randomUnitPos;
            totalDensity += GetDensity(randomSamplePos);
        }
        return totalDensity;
    }

    void Tree::SetupDebug() {
        debugShader.InitVertexShaderFromFile("../shaders/tree-debug.vert");
        debugShader.InitFragmentShaderFromFile("../shaders/tree-debug.frag");
        debugShader.Link();

        vao->Bind();
        vao->EnableAttributeArray(0);
        vao->SpecifyAttributeArrayFloat(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0, vbo.get());

        vbo->Bind(GL_ARRAY_BUFFER);
        vbo->BufferData(BOX_VERTS.size() * sizeof(float), BOX_VERTS.data(), GL_STATIC_DRAW);

        ibo->Bind(GL_ELEMENT_ARRAY_BUFFER);
        ibo->BufferData(BOX_INDICES.size() * sizeof(uint32_t), BOX_INDICES.data(), GL_STATIC_DRAW);
    }

    void Tree::SetDebug(bool state) {
        debugMode = state;
    }

    void Tree::DrawDebug() {

    }
}
