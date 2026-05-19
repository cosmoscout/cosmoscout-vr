#include "Tree.hpp"

namespace csp::atmospheres {
    Tree::Tree(glm::vec3 minBounds, glm::vec3 maxBounds) {
        builtNodes = std::vector<Node>();
        this->minBounds = minBounds;
        this->maxBounds = maxBounds;

        shader = std::make_unique<VistaGLSLShader>();
        if (!shader->InitComputeShaderFromFile("../share/resources/shaders/octree.comp"))
            vstr::debug() << "Failed to init compute shader in atmosphere program." << std::endl;
        utils::storeShaderInfoLog("octree.comp", shader->GetComputeShader(0));
        shader->Link();

        shader->Bind();
        // 1. Setup nodes buffer
        glGenBuffers(1, &nodesBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, nodesBuffer);
        // No nodes array initialisation required?
        glBufferData(GL_SHADER_STORAGE_BUFFER, MAX_NODE_SIZE, nullptr, GL_DYNAMIC_DRAW);

        // 2. Initialize root node (Index 0)
        Node root;
        root.boundsMin = minBounds;
        root.boundsMax = maxBounds;
        root.isLeaf = 1;
        root.depth = 0;
        for(int i = 0; i < 8; ++i)
            root.children[i] = -1;

        // Upload the root node to offset 0
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(Node), &root);

        // 3. Setup work queue buffer
        glGenBuffers(1, &queueBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, queueBuffer);

        // Initialize queue
        int initialQueue[MAX_QUEUE];
        // Push root node as work task
        initialQueue[0] = 0;
        // Fill rest with 0s
        for(int i = 1; i < MAX_QUEUE; ++i)
            initialQueue[i] = 0;

        glBufferData(GL_SHADER_STORAGE_BUFFER, MAX_QUEUE_SIZE, initialQueue, GL_DYNAMIC_DRAW);
        
        // Initialize atomics (Head = 0, Tail = 1 because we have 1 item, the root node)
        headCount = 0;
        tailCount = 1;
        unsigned int atomicVals[] = { headCount, tailCount };
        glGenBuffers(1, &atomicsBuffer);
        // Bind the buffer and define its initial storage capacity
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicsBuffer);
        glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint) * 2, atomicVals, GL_DYNAMIC_DRAW);
        // Unbind the buffer        
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);

        // Set bounds
        float _minBounds[] = { minBounds[0], minBounds[1], minBounds[2] };
        float _maxBounds[] = { maxBounds[0], maxBounds[1], maxBounds[2] };
        glUniform3fv(MIN_BOUNDS_LOC, 3, _minBounds);
        glUniform3fv(MAX_BOUNDS_LOC, 3, _maxBounds);
    }

    void Tree::Build() {
        // Bind the SSBOs to the shader bindings (0 = nodes, 1 = queue)
        shader->Bind();

        // Bind the allocated buffer memory at ID atomicsBuffer, ... to the binding ATOMICS_BUFFER_BINDING used on the compute shader 
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, NODE_BUFFER_BINDING, nodesBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, QUEUE_BUFFER_BINDING, queueBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ATOMICS_BUFFER_BINDING, atomicsBuffer);

        // --- THE LOOP ---
        bool complete = false;
        
        while (!complete) {
            // 1. Dispatch Work (e.g., 16 groups of 256 threads = 4096 threads)
            glDispatchCompute(32, 1, 1);
            
            // 2. Barrier to ensure shader writes (atomic adds) are visible
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);

            // 3. Read back the atomic counters from the GPU
            UpdateCounts();

            if (headCount % 100 == 0)
                vstr::debug() << "Built " << headCount << " nodes, " << tailCount << " left." << std::endl;
            // We read from the buffer binding point (which is 1)
            // Note: In a real loop, reading buffer data is expensive. 
            // For simple trees this is fine. For massive trees, consider a different topology.

            // glBindBuffer(GL_PIXEL_PACK_BUFFER, 0); // Unbind PBO if used, or use GL_READ_BUFFER
            
            // 4. Read back the Head counter (optional, strictly speaking Tail is enough for "is there work?")
            // If Tail == Head, the queue is empty.
            complete = tailCount == 0;
        }

        FetchNodes();
        vstr::debug() << "Built octree with size = " << headCount << std::endl;
    }

    void Tree::FetchNodes() {
        if (headCount == builtNodes.size()) {
            vstr::debug() << "Head count " << headCount << " unchanged; aborting builtNodes update." << std::endl;
        }

        Node nodes[MAX_NODES];
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, nodesBuffer);
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, headCount * sizeof(Node), nodes);

        builtNodes.clear();
        for (size_t i = 0; i < headCount; i++) {
            builtNodes.push_back(nodes[i]);
        }
    }

    const char *TREE_DEBUG_VERT_SHADER = R"(#version 400
        layout (location = 0) in vec3 inPos;
        out vec3 pos;

        uniform mat4 modelViewMat;
        uniform mat4 projMat;

        uniform float maxBounds;

        uniform vec3 aabbMin;
        uniform vec3 aabbMax;

        // const mat3 isoMat = mat3(
        //         sqrt(3), -1, sqrt(2),
        //         0, 2, sqrt(2),
        //         -sqrt(3), -1, sqrt(2)
        //     ); 

        void main() {
            // Isometric projection
            // vec3 projected = isoMat * (inPos * (aabbMax - aabbMin) + aabbMin);
            // projected /= 1000 * 1000 * 50;
            // projected.xy /= (1.0 + projected.z);

            pos = inPos * (aabbMax - aabbMin) + aabbMin;
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
                float densityScale = remap(density, 0.2, 0.5, 0, 1);
                color = vec4(0, 1, densityScale, 1);
            } else {
                color = vec4(1, 0.2, 0, 1);
            }
            // gl_FragDepth = length(pos);
        })";

    void Tree::SetupDebug(float maxBoundsAxis) {
        debugShader->InitVertexShaderFromString(TREE_DEBUG_VERT_SHADER);
        debugShader->InitFragmentShaderFromString(TREE_DEBUG_FRAG_SHADER);
        debugShader->Link();

        debugShader->Bind();
        debugShader->SetUniform(debugShader->GetUniformLocation("maxBounds"), maxBoundsAxis);
        debugShader->Release();

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

    void Tree::DrawDebug(const glm::mat4 &modelViewMat, const glm::mat4 &projMat) {
        if (debugMode) {
            glPushAttrib(GL_POLYGON_BIT);
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

            debugShader->Bind();
            vao->Bind();

            glUniformMatrix4fv(debugShader->GetUniformLocation("modelViewMat"), 1, GL_FALSE, glm::value_ptr(modelViewMat));
            glUniformMatrix4fv(debugShader->GetUniformLocation("projMat"), 1, GL_FALSE, glm::value_ptr(projMat));            

            for (size_t i = 0; i < builtNodes.size(); i++) {
                const auto &node = builtNodes[i];
                if (!node.isLeaf)
                    continue;
                
                debugShader->SetUniform(debugShader->GetUniformLocation("aabbMin"), node.boundsMin[0], node.boundsMin[1], node.boundsMin[2]);
                debugShader->SetUniform(debugShader->GetUniformLocation("aabbMax"), node.boundsMax[0], node.boundsMax[1], node.boundsMax[2]);
                debugShader->SetUniform(debugShader->GetUniformLocation("density"), (float)node.depth);

                glDrawElements(GL_LINES, 24, GL_UNSIGNED_INT, 0);
            }

            debugShader->Release();
            vao->Release();

            glPopAttrib();
        }
    }
}