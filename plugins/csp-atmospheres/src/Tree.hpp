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
    typedef struct OctreeNode {
        int children[8] = { -1, 0, 0, 0, 0, 0, 0, 0 };
        glm::vec3 boundsMin;
        glm::vec3 boundsMax;
        unsigned int depth = 0;
    } Node;

    bool IsLeafNode(const Node &node) {
        return node.children[0] == -1;
    }

    const int MAX_NODES = 1 << 16;
    const int MAX_QUEUE = 1 << 16;
    const int MAX_NODE_SIZE = MAX_NODES * sizeof(Node);
    const int MAX_QUEUE_SIZE = MAX_QUEUE * sizeof(unsigned int);
    const unsigned int ROOT_NODE_INDEX = 0;

    const int NODE_BUFFER_BINDING = 0;
    const int QUEUE_BUFFER_BINDING = 1;
    const int ATOMICS_BUFFER_BINDING = 2;
    const int MIN_BOUNDS_LOC = 0;
    const int MAX_BOUNDS_LOC = 1;

    const float CUMULONIMBUS_START_HEIGHT = 1500;
    const float CUMULONIMBUS_END_HEIGHT = 5000;

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
        glm::vec3 minBounds, maxBounds;

        std::unique_ptr<VistaGLSLShader> shader;
        GLuint nodesBuffer, queueBuffer, atomicsBuffer;

        std::vector<Node> builtNodes;
        unsigned int headCount, tailCount;

        // -- DEBUG --
        bool debugMode = true;
        std::unique_ptr<VistaGLSLShader> debugShader;
        std::unique_ptr<VistaBufferObject> vbo, ibo;
        std::unique_ptr<VistaVertexArrayObject> vao;

        void SetupDebug(float maxBoundsAxis);

        void UpdateCounts() {
            GLuint counters[2];
            glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, atomicsBuffer);
            glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint) * 2, counters);
            glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0);
            headCount = counters[0];
            tailCount = counters[1];
        }

    public:
        Tree();
        void Setup(const glm::vec3 &minBounds, const glm::vec3 &maxBounds);
        void Build();

        void SetDebug(bool state) {
            debugMode = state;
        }

        bool GetDebug() const {
            return debugMode;
        }

        void DrawDebug(const glm::mat4 &modelViewMat, const glm::mat4 &projMat);

        float GetMaxSize() const {
            return glm::length(maxBounds);
        }

        void FetchNodes();
    };
}

#endif