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

    unsigned int Tree::Subdivide(unsigned int index, unsigned int depth, bool check) {
        if (debugMode && index > 0 && index % (maxNodeCount / 100) == 0)
            vstr::debug() << "Subdivided " << ((float)index / maxNodeCount) * 100.0f << "% of max nodes."  << std::endl;

#ifdef TREE_DEBUG_MODE
        if (debugMode) {
            for (size_t i = 0; i < depth; i++)
            {
                vstr::debug() << " ";
            }
            vstr::debug() << "[Depth = " << depth << "] index " << index << ": ";
        }
#endif

        auto &node = nodes[index];

        float totalDensity = 0.0f;
        const float DENSITY_SAMPLING_DECREASE_EXP = 0.33f;
        totalDensity = GetTotalDensity(index, depth, (unsigned int)(BASE_DENSITY_SAMPLES / pow(depth + 1, DENSITY_SAMPLING_DECREASE_EXP)));
        // TEMP: fix shallow subdivision by forcing octree to divide if depth <= maxDepth / 2
        if (totalDensity < MIN_DENSITY_CUTOFF) {
#ifdef TREE_DEBUG_MODE
            if (debugMode)
                vstr::debug() << "zero density. STOP" << std::endl;
#endif
            return 0;
        } else {
#ifdef TREE_DEBUG_MODE
            if (debugMode)
                vstr::debug() << "density = " << totalDensity << ". ";
#endif
        }
        node.density = totalDensity;

        if (depth >= maxDepth) { // If level of depth has been reached, stop subdivision process.
#ifdef TREE_DEBUG_MODE
        if (debugMode)
            vstr::debug() << "max depth reached. STOP" << std::endl;
#endif
            return 0;
        }
        
#ifdef TREE_DEBUG_MODE
        if (debugMode)
            vstr::debug() << std::endl;
#endif

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
        if (glm::length(centre) > 1e-4) // If node is not root (i.e., aabbMin == -aabbMax, thus linearly dependent and centre == zero vector)...
            centre = node.aabbMin + centre; // ...then convert from direction to position
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
        }
        node.childrenCount = children;
        return children;
    }

    float Tree::GetDensity(glm::vec3 pos) {
        glm::vec2 density = GetCloudDensity(pos, properties);
        return density.x;
    }

    float Tree::GetTotalDensity(unsigned int index, unsigned int depth, unsigned int totalSamples = BASE_DENSITY_SAMPLES) {
        // IDEA: dont sample at random positions, but find where the cloud layer is inside this bounding box
        // and sample there. Much more precise and possible due to aabbMin, aabbMax being absolute positions.

        auto &node = nodes[index];

        // Check intersection of node with outer cloud layer.
        // (mode = 2: solid box, hollow sphere => if box is inside sphere, no hit registered, other way around is a hit)
        // bool intersectOuterClouds = IntersectAABBSphere(node.aabbMin, node.aabbMax, properties.planetRadius + CUMULONIMBUS_END_HEIGHT, 2);
        // if (!intersectOuterClouds) { // Node is outside of completely inside or outside outer cloud layer.
        //     vstr::debug() << "[Depth " << depth << "] Node " << glm::to_string(node.aabbMin / 1000.0f) << ", "
        //         << glm::to_string(node.aabbMax / 1000.0f) << " no intersect with outer cloud layer (" << (properties.planetRadius + CUMULONIMBUS_END_HEIGHT) / 1000.0f << ")" << std::endl;
        //     // Check if node intersects inner layer at least.
        //     bool intersectInnerCloudLayer = IntersectAABBSphere(node.aabbMin, node.aabbMax, properties.planetRadius + CUMULONIMBUS_START_HEIGHT, 2);
        //     if (!intersectInnerCloudLayer) {
        //         vstr::debug() << "[Depth " << depth << "] No intersect with inner cloud layer (" << (properties.planetRadius + CUMULONIMBUS_START_HEIGHT) / 1000.0f << ")" << std::endl;
        //         return 0.0f;
        //     }
        // }
        static glm::vec3 origin(0.0f);
        bool intersectOuterCloudLayer = IntersectAABBSphere(node.aabbMin, node.aabbMax, properties.planetRadius + CUMULONIMBUS_START_HEIGHT, origin);
        bool intersectInnerCloudLayer = IntersectAABBSphere(node.aabbMin, node.aabbMax, properties.planetRadius + CUMULONIMBUS_END_HEIGHT, origin);
        if (!intersectInnerCloudLayer && !intersectOuterCloudLayer) {
            // if (depth <= 3)
            //     vstr::debug() << "Nodes[" << index << "] at depth " << depth << " not intersecting cloud layer: aborting." << std::endl;
            return 0.0f;
        }

        // Sample random positions inside the bounding box
        static std::random_device rd;  // Will be used to obtain a seed for the random number engine
        static std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
        static std::uniform_real_distribution<float> distr(0.0f, 1.0f);

        glm::vec3 extends;
        if (index == 0) // Fix root node issue: rootNode.aabbMin == -rootNode.aabbMax => rootNode centre == zero vector (linearly dependent)
            extends = node.aabbMax * 2.0f;
        else
            extends = node.GetExtends();
        
        for (unsigned int i = 0; i < totalSamples; i++) {
            // Generate random position in [0, 1) x [0, 1) x [0, 1)
            glm::vec3 randomPos(distr(gen), distr(gen), distr(gen));
            glm::vec3 samplePos = node.aabbMin + extends * randomPos;
            float density = GetDensity(samplePos);
            if (density > 0.0f)
                return density;
        }

        return 0.0f;
    }

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

        float remap(float t, float minOld, float maxOld, float minNew, float maxNew) {
            float tRescaled = (t - minOld) / (maxOld - minOld);
            return clamp(tRescaled * (maxNew - minNew) + minNew, min(minNew, maxNew), max(maxNew, minNew));
        }

        void main() {
            if (density > 1e-6) {
                float densityScale = remap(density, 1e-6, 0.5, 1e-6, 1);
                color = vec4(0, 1, densityScale, 1);
            } else {
                color = vec4(1, 0.2, 0, 1);
            }
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
                if (!node.IsLeaf())
                    continue;
                
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
