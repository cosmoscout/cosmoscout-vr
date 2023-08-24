////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "LineRenderer.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/Settings.hpp"

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

    const char* LineRenderer::FEATURE_VERT = R"(
    #version 330 core
    layout (location=0) in vec3 aPos;
    layout (location=1) in vec3 aColor;

    out vec3 vertexColor;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main(){
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        vertexColor = aColor;
    }
    )";

    const char* LineRenderer::FEATURE_FRAG = R"(

    in vec3 vertexColor;

    uniform float         uAmbientBrightness;
    uniform float         uSunIlluminance;
    uniform vec3          uSunDirection;

    out vec4 FragColor;

    const float PI = 3.14159265359;
    // ===========================================================================
    vec3 SRGBtoLINEAR(vec3 srgbIn)
    {
        vec3 bLess = step(vec3(0.04045),srgbIn);
        return mix( srgbIn/vec3(12.92), pow((srgbIn+vec3(0.055))/vec3(1.055),vec3(2.4)), bLess );
    }

    void main(){

        vec3 result = vertexColor.rgb;

            #ifdef ENABLE_HDR
                result = SRGBtoLINEAR(result) * uSunIlluminance / PI;
            #else
                result = result * uSunIlluminance;
            #endif

            FragColor = vec4(result, 1.0);
    }
    )";

    LineRenderer::LineRenderer (std::vector<glm::vec3> coordinates, std::shared_ptr<cs::core::SolarSystem> solarSystem,   
                                        std::shared_ptr<cs::core::Settings> settings, double lineWidth, std::shared_ptr<Settings> pluginSettings) {
        mSolarSystem    = solarSystem;
        mCoordinates    = coordinates;
        mSettings       = settings;
        mPluginSettings = pluginSettings;
        mLineWidthInput = lineWidth;
        mShaderDirty    = true;

        VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
        mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
        VistaOpenSGMaterialTools::SetSortKeyOnSubtree(mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eOpaqueItems));
        
        glGenVertexArrays(1, &mVAO);

        // Generating the VBO to manage the memory of the input vertex
        unsigned int VBO;                                                                   
        glGenBuffers(1, &VBO);

        // Binding the mVAO
        glBindVertexArray(mVAO);                                                             
        // Binding the VBO
        glBindBuffer(GL_ARRAY_BUFFER, VBO); 

        // Copying user-defined data into the current bound buffer                                                                                                        
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * mCoordinates.size(), mCoordinates.data(), GL_STATIC_DRAW);      

        // Setting the vertex coordinates attribute pointers
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);       
        glEnableVertexAttribArray(0);

        // Setting the vertex colors attributes pointers
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_TRUE, 6 * sizeof(float), (void*)(3 * sizeof(float)));       // color vertex attribute pointers
        glEnableVertexAttribArray(1);

        glBindBuffer(GL_ARRAY_BUFFER, 0); 
        glBindVertexArray(0);

        // Recreate the shader if HDR rendering mode are toggled.
        mHDRConnection = mSettings->mGraphics.pEnableHDR.connect([this](bool /*unused*/) { mShaderDirty = true; });
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    LineRenderer::~LineRenderer() {
    
        mSettings->mGraphics.pEnableHDR.disconnect(mHDRConnection);

        VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
        pSG->GetRoot()->DisconnectChild(mGLNode.get());
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    bool LineRenderer::Do() {

        if (!mPluginSettings->mEnabled.get()) {
            return true;
        }

        if (mShaderDirty) {

            // building and compiling the vertex Shader
            //-----------------------------------------
            unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
            glShaderSource(vertexShader, 1, &FEATURE_VERT, NULL);
            glCompileShader(vertexShader);
            // check for shader compile errors
            int success;
            char infoLog[512];
            glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
            if (!success) {
                glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
                std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
            }
            // HDR settings
            std::string defines = "#version 330\n";
            if (mSettings->mGraphics.pEnableHDR.get()) {
                defines += "#define ENABLE_HDR\n";
            }
            std::string fragmentShaderCode = defines + FEATURE_FRAG;
            const char * fragmentShadercStr = fragmentShaderCode.c_str();

            // building and compiling the fragment Shader
            //-------------------------------------------
            unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
            glShaderSource(fragmentShader, 1, &fragmentShadercStr, NULL);
            glCompileShader(fragmentShader);
            // check for shader compile errors
            glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
            if (!success) {
                glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
                std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
                logger().info("{}", defines + FEATURE_FRAG);
            }

            // linking shaders
            //----------------
            mShaderProgram = glCreateProgram();
            glAttachShader(mShaderProgram, vertexShader);
            glAttachShader(mShaderProgram, fragmentShader);
            glLinkProgram(mShaderProgram);
            // check for linking errors
            glGetProgramiv(mShaderProgram, GL_LINK_STATUS, &success);
            if (!success) {
                glGetProgramInfoLog(mShaderProgram, 512, NULL, infoLog);
                std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
            }
            glDeleteShader(vertexShader);
            glDeleteShader(fragmentShader);
            mShaderDirty = false;
        }

        auto earth = mSolarSystem->getObject("Earth");
        if (!earth || !earth->getIsBodyVisible()) {
            return true;
        }

        glUseProgram(mShaderProgram);
        auto transform = earth->getObserverRelativeTransform();

        glEnable(GL_PROGRAM_POINT_SIZE);

        // Dealing with the different uniform matrices.
        //---------------------------------------------
        std::array<GLfloat, 16> glMatV{};
        std::array<GLfloat, 16> glMatP{};
        glGetFloatv(GL_MODELVIEW_MATRIX, glMatV.data());
        glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());
        auto matM = glm::mat4(transform);
        auto matV = glm::make_mat4x4(glMatV.data());
        // Getting the uniform matrices location
        GLuint model = glGetUniformLocation(mShaderProgram, "model");
        GLuint view = glGetUniformLocation(mShaderProgram, "view");
        GLuint projection = glGetUniformLocation(mShaderProgram, "projection");
        // Doing the uniform matrices location assignment
        glUniformMatrix4fv(model, 1, GL_FALSE, glm::value_ptr(matM));
        glUniformMatrix4fv(view, 1, GL_FALSE, glm::value_ptr(matV));
        glUniformMatrix4fv(projection, 1, GL_FALSE, glMatP.data());

        // HDR settings
        //-------------
        glm::vec3 sunDirection(1, 0, 0);
        float sunIlluminance(1.F);
        float ambientBrightness(mSettings->mGraphics.pAmbientBrightness.get());

        if (mSettings->mGraphics.pEnableHDR.get()) {
            sunIlluminance = static_cast<float>(mSolarSystem->getSunIlluminance(transform[3]));
        }
        sunDirection = glm::normalize(glm::inverse(transform) * glm::dvec4(mSolarSystem->getSunDirection(transform[3]), 0.0));
        // HDR uniform locations
        GLint uSunDirectionLocation  = glGetUniformLocation(mShaderProgram, "uSunDirection");
        GLint uSunIlluminanceLocation  = glGetUniformLocation(mShaderProgram, "uSunIlluminance");
        GLint uAmbientBrightnessLocation  = glGetUniformLocation(mShaderProgram, "uAmbientBrightness");
        // HDR uniform assignment
        glUniform1f (uAmbientBrightnessLocation, ambientBrightness);
        glUniform1f (uSunIlluminanceLocation, sunIlluminance);
        glUniform3fv (uSunDirectionLocation, 1, glm::value_ptr(sunDirection));

        // Draw
        glBindVertexArray(mVAO); 
        GLfloat lineWidthGL = static_cast<GLfloat>(mLineWidthInput);
        glLineWidth(lineWidthGL);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(3 * mCoordinates.size()));
        glBindVertexArray(0); 

        return true;
    }

    bool LineRenderer::GetBoundingBox(VistaBoundingBox& bb) {
        return false;
    }
}
