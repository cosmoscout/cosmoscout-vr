////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "FeatureRenderer.hpp"
#include "logger.hpp"

#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaMath/VistaBoundingBox.h>
#include <VistaOGLExt/VistaOGLUtils.h>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "../../../src/cs-utils/utils.hpp"

namespace csp::wfsoverlays {

    const char* FeatureRenderer::FEATURE_VERT = R"(
    #version 330 core
    layout (location=0) in vec3 aPos;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main(){
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }
    )";

    const char* FeatureRenderer::FEATURE_FRAG = R"(
    #version 330 core

    out vec4 FragColor;

    void main(){
        FragColor = vec4(1.0f, 0.5f, 1.0f, 1.0f);
    }
    )";

    FeatureRenderer::FeatureRenderer(WFSFeatureCollection collection, std::shared_ptr<cs::core::SolarSystem> solarSystem) {
        
        mSolarSystem = solarSystem;
        auto earth = mSolarSystem->getObject("Earth");
        glm::dvec3 earthRadius = earth->getRadii();
        
        for (int i=0; i < collection.totalFeatures; i++) {
            double phiRadians = ((collection.features[i].geometry.coordinates[0])*2*glm::pi<double>())/360.0;
            double thetaRadians  = ((90-collection.features[i].geometry.coordinates[1])*2*glm::pi<double>())/360.0;

            glm::dvec3 normalizedCoords = {sin(thetaRadians)*sin(phiRadians), 
                                    cos(thetaRadians),
                                    sin(thetaRadians)*cos(phiRadians)                                
                                    };                                   
            glm::vec3 cartesianCoordinates = normalizedCoords * earthRadius;
            coordinates.push_back(cartesianCoordinates); 
        };
        
        /* logger().info("First dataset");
        logger().info(cartesianCoordinates[0]);
        logger().info(cartesianCoordinates[1]);
        logger().info(cartesianCoordinates[2]); */

        VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
        mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
        VistaOpenSGMaterialTools::SetSortKeyOnSubtree(mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eOpaqueItems));
        
        // building and compiling the vertex Shader
        //--------------
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShader, 1, &FEATURE_VERT, NULL);
        glCompileShader(vertexShader);
        // check for shader compile errors
        int success;
        char infoLog[512];
        glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
        }

        // building and compiling the fragment Shader
        //----------------
        unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShader, 1, &FEATURE_FRAG, NULL);
        glCompileShader(fragmentShader);
        // check for shader compile errors
        glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
        }

        // linking shaders
        //-------------
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);
        // check for linking errors
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if (!success) {
            glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
            std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
        }
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        glGenVertexArrays(1, &VAO);

        unsigned int VBO;                                                                   // Generating the VBO (vertex Buffer Object), which manages the memory storage of the input vertex
        glGenBuffers(1, &VBO);

        glBindVertexArray(VAO);                                                             // Binding the VAO

        glBindBuffer(GL_ARRAY_BUFFER, VBO);                                                 // Binding the VBO                                                        
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * coordinates.size(), coordinates.data(), GL_STATIC_DRAW);          // Copying user-defined data into the current bound buffer

        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);       // Setting the vertex attributes pointers
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER, 0); 
        glBindVertexArray(0);
    }

    glm::vec3 FeatureRenderer::getCoordinates(int i) const {
        return coordinates[i];   
    }

    bool FeatureRenderer::Do() {
        auto earth = mSolarSystem->getObject("Earth");
        if (!earth || !earth->getIsBodyVisible()) {
            return true;
        }

        glUseProgram(shaderProgram);
        auto transform = earth->getObserverRelativeTransform();

        // Get modelview and projection matrices.
        std::array<GLfloat, 16> glMatV{};
        std::array<GLfloat, 16> glMatP{};
        glGetFloatv(GL_MODELVIEW_MATRIX, glMatV.data());
        glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
        auto matM = glm::mat4(transform);
        auto matV = glm::make_mat4x4(glMatV.data());

        GLuint model = glGetUniformLocation(shaderProgram, "model");
        GLuint view = glGetUniformLocation(shaderProgram, "view");
        GLuint projection = glGetUniformLocation(shaderProgram, "projection");

        glUniformMatrix4fv(model, 1, GL_FALSE, glm::value_ptr(matM));
        glUniformMatrix4fv(view, 1, GL_FALSE, glm::value_ptr(matV));
        glUniformMatrix4fv(projection, 1, GL_FALSE, glMatP.data());

        // draw our first triangle
        glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized
        glPointSize(5.0);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(3 * coordinates.size()));
        glBindVertexArray(0); // no need to unbind it every time 
        return true;
    }

    bool FeatureRenderer::GetBoundingBox(VistaBoundingBox& bb) {
        return false;
    }
}
