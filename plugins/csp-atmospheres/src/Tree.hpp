////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_TREE_BVH_HPP
#define CSP_TREE_BVH_HPP

// #define TREE_DEBUG_MODE
// #define TREE_DEBUG_HIDE_EMPTY_NODES

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <memory>
#include <algorithm>
#include <vector>
#include <math.h>
#include <cmath>
#include <array>
#include <VistaBase/VistaStreamUtils.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

#include "utils.hpp"
#include <random>

#include <limits>

namespace csp::atmospheres {
    // Define the Node struct to match GLSL exactly
    typedef struct {
        int children[8];
        glm::vec3 boundsMin;
        glm::vec3 boundsMax;
        unsigned int isLeaf;
        unsigned int depth;
    } Node;

    const int MAX_NODES = 1 << 16;
    const int MAX_QUEUE = 1 << 16;

    // Wireframe nodes used in debug mode: local coordinates not exactly at 0/ 1 to introduce a small padding at the edges.
    const std::array BOX_VERTS = {
        /*0*/ 0.001F, 0.001F, 0.001F, /*1*/ 0.001F, 0.001F, 0.999F, /*2*/ 0.001F, 0.999F, 0.001F, /*3*/ 0.001F, 0.999F, 0.999F,
        /*4*/ 0.999F, 0.001F, 0.001F, /*5*/ 0.999F, 0.001F, 0.999F, /*6*/ 0.999F, 0.999F, 0.001F, /*7*/ 0.999F, 0.999F, 0.999F
    };
    const std::array BOX_INDICES = {
        0U, 1U, 0U, 2U, 0U, 4U, 1U, 3U, 1U, 5U, 2U, 3U, 2U, 6U, 3U, 7U, 4U, 5U, 4U, 6U, 5U, 7U, 6U, 7U
    };

    class Tree {
    private:
        const size_t queueOffset = MAX_QUEUE * sizeof(unsigned int);

        GLuint shaderProgram, computeShader;
        GLuint nodesBuffer;
        GLuint queueBuffer;

        std::vector<Node> builtNodes;

        // -- DEBUG --
        bool debugMode = true;
        std::unique_ptr<VistaGLSLShader> debugShader;
        std::unique_ptr<VistaBufferObject> vbo, ibo;
        std::unique_ptr<VistaVertexArrayObject> vao;

        void SetupDebug(float maxBoundsAxis);

    public:

        Tree(VistaGLSLShader &atmosphereShader, glm::vec3 minBounds, glm::vec3 maxBounds);
        void Build();

        void SetDebug(bool state) {
            debugMode = state;
        }

        bool GetDebug() const {
            return debugMode;
        }

        void DrawDebug(const glm::mat4 &modelViewMat, const glm::mat4 &projMat);

        unsigned int GetHeadCount() const {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, queueBuffer);
            unsigned int headCount = 0;
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, queueOffset, sizeof(unsigned int), &headCount);
            return headCount;
        }

        unsigned int GetTailCount() const {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, queueBuffer);
            unsigned int tailCount = 0;
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, queueOffset + sizeof(unsigned int), sizeof(unsigned int), &tailCount);
            return tailCount;
        }

        void FetchNodes() {
            unsigned int headCount = GetHeadCount();
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
    };
}

#endif