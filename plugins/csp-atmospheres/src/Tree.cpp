#include "Tree.hpp"

namespace csp::atmospheres {
    Tree::Tree(VistaGLSLShader &atmosphereShader, glm::vec3 minBounds, glm::vec3 maxBounds) {
        builtNodes = std::vector<Node>();

        atmosphereShader.InitComputeShaderFromFile("../share/resources/shaders/octree.comp");
        computeShader = atmosphereShader.GetComputeShader(0);
        utils::storeShaderInfoLog("octree.comp", computeShader);

        // 1. Setup Nodes Buffer
        glGenBuffers(1, &nodesBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, nodesBuffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Node) * MAX_NODES, nullptr, GL_DYNAMIC_DRAW);

        // 2. Initialize Root Node (Index 0)
        Node root;
        root.boundsMin = minBounds;
        root.boundsMax = maxBounds;
        root.isLeaf = 1;
        root.depth = 0;
        for(int i=0; i<8; ++i) root.children[i] = -1;

        // Upload the root node to offset 0
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(Node), &root);

        // 3. Setup Work Queue Buffer
        glGenBuffers(1, &queueBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, queueBuffer);
        
        // The queue array (int) + Head (uint) + Tail (uint)
        // Calculate offsets for the atomics based on array size
        size_t headOffset = queueOffset;
        size_t tailOffset = queueOffset + sizeof(unsigned int);
        size_t totalSize = tailOffset + sizeof(unsigned int);

        glBufferData(GL_SHADER_STORAGE_BUFFER, totalSize, nullptr, GL_DYNAMIC_DRAW);

        // Initialize Queue: Push Root Index (0)
        int initialQueue[MAX_QUEUE];
        initialQueue[0] = 0; // The root node
        // Fill rest with 0s
        for(int i=1; i<MAX_QUEUE; ++i) initialQueue[i] = 0;
        
        // Initialize Atomics (Head = 0, Tail = 1 because we have 1 item)
        unsigned int headVal = 0;
        unsigned int tailVal = 1;

        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, queueOffset, initialQueue);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, headOffset, sizeof(unsigned int), &headVal);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, tailOffset, sizeof(unsigned int), &tailVal);
    }

    void Tree::Build() {
        // Bind the SSBOs to the shader bindings (0=Nodes, 1=Queue)
        glUseProgram(shaderProgram);

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, nodesBuffer);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, queueBuffer);

        // --- THE LOOP ---
        bool isComplete = false;
        
        while (!isComplete) {
            // 1. Dispatch Work (e.g., 16 groups of 256 threads = 4096 threads)
            glDispatchCompute(16, 1, 1);
            
            // 2. Barrier to ensure shader writes (atomic adds) are visible
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT);

            // 3. Read back the Tail counter from the GPU
            GLuint tailVal = 0;
            // We read from the buffer binding point (which is 1)
            // Note: In a real loop, reading buffer data is expensive. 
            // For simple trees this is fine. For massive trees, consider a different topology.
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0); // Unbind PBO if used, or use GL_READ_BUFFER
            
            // Direct readback (Blocking call, but safe here)
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, queueOffset + sizeof(unsigned int), sizeof(GLuint), &tailVal);
            vstr::debug() << "Octree build: tail counter = " << tailVal << std::endl;
            
            // 4. Read back the Head counter (optional, strictly speaking Tail is enough for "is there work?")
            // If Tail == Head, the queue is empty.
            isComplete = tailVal == 0;
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

            for (size_t i = 0; i < GetTailCount(); i++) {
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

    void Tree::DrawWireframe() {

    }
}