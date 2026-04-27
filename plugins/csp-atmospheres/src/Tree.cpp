////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Tree.hpp"

namespace csp::atmospheres {
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
        debugShader = std::make_unique<VistaGLSLShader>();
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
                vstr::debug() << "max depth reached. STOP" << std::endl;

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

    // MV Matrix immer relativ zum Planeten der betrachtet wird. MV-Matrix: ObserverRelativeTransform
    // https://github.com/cosmoscout/cosmoscout-vr/blob/d26a33b6b706126abf9faae8aa6056fcde1892ba/plugins/csp-simple-bodies/src/SimpleBody.cpp#L458 

    const char *TREE_DEBUG_VERT_SHADER = R"(#version 400
        layout (location = 0) in vec3 inPos;
        out vec3 pos;

        uniform mat4 modelViewMat;
        uniform mat4 projMat;

        uniform float maxBounds;

        uniform vec3 aabbMin;
        uniform vec3 aabbMax;

        const mat3 isoMat = mat3(
                sqrt(3), -1, sqrt(2),
                0, 2, sqrt(2),
                -sqrt(3), -1, sqrt(2)
            ); 

        void main() {
            // Isometric projection
            // vec3 projected = isoMat * (inPos * (aabbMax - aabbMin) + aabbMin);
            // projected /= 1000 * 1000 * 50;
            // projected.xy /= (1.0 + projected.z);

            vec3 pos = inPos * (aabbMax - aabbMin) + aabbMin;
            pos /= maxBounds; // Scale to [0, 1] x ... x [0, 1] screen space
            vec4 projPos = projMat * modelViewMat * vec4(pos, 1);
            gl_Position = projPos;
        })";
    const char *TREE_DEBUG_FRAG_SHADER = R"(#version 400
        in vec3 pos;
        out vec4 color;

        uniform float density;

        void main() {
            float scaledDensity = density / 8;
            float clampedDensity = clamp(scaledDensity, 0, 1);
            color = vec4(clampedDensity, 0, 1, density > 0.0 ? 1 : 0);
            // gl_FragDepth = length(pos);
        })";

    void Tree::SetupDebug() {
        debugShader->InitVertexShaderFromString(TREE_DEBUG_VERT_SHADER);
        debugShader->InitFragmentShaderFromString(TREE_DEBUG_FRAG_SHADER);
        debugShader->Link();

        vao->Bind();
        vao->EnableAttributeArray(0);
        vao->SpecifyAttributeArrayFloat(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0, vbo.get());

        vbo->Bind(GL_ARRAY_BUFFER);
        vbo->BufferData(BOX_VERTS.size() * sizeof(float), BOX_VERTS.data(), GL_STATIC_DRAW);

        ibo->Bind(GL_ELEMENT_ARRAY_BUFFER);
        ibo->BufferData(BOX_INDICES.size() * sizeof(uint32_t), BOX_INDICES.data(), GL_STATIC_DRAW);

        vao->Release();
        vbo->Release();
        ibo->Release();
    }

    void Tree::SetDebug(bool state) {
        debugMode = state;
    }

    void Tree::DrawDebug(const glm::mat4 &modelViewMat, const glm::mat4 &projMat) {
        glPushAttrib(GL_POLYGON_BIT);
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        if (debugMode) {
            debugShader->Bind();
            vao->Bind();

            glUniformMatrix4fv(debugShader->GetUniformLocation("modelViewMat"), 1, GL_FALSE, glm::value_ptr(modelViewMat));
            glUniformMatrix4fv(debugShader->GetUniformLocation("projMat"), 1, GL_FALSE, glm::value_ptr(projMat));
            
            
            debugShader->SetUniform(debugShader->GetUniformLocation("maxBounds"), GetMaxBounds());

            for (size_t i = 0; i < GetUsedNodeCount(); i++) {
                const auto &node = nodes[i];
                debugShader->SetUniform(debugShader->GetUniformLocation("aabbMin"), node.aabbMin[0], node.aabbMin[1], node.aabbMin[2]);
                debugShader->SetUniform(debugShader->GetUniformLocation("aabbMax"), node.aabbMax[0], node.aabbMax[1], node.aabbMax[2]);
                debugShader->SetUniform(debugShader->GetUniformLocation("density"), node.density);

                glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);
            }

            debugShader->Release();
            vao->Release();
        }

        glPopAttrib();
    }
}
