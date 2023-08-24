////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "PointRenderer.hpp"
#include "logger.hpp"

#include "../../../src/cs-core/Settings.hpp"

#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/GraphicsManager/VistaTransformNode.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaMath/VistaBoundingBox.h>
#include <VistaOGLExt/VistaOGLUtils.h>
#include "../../../src/cs-graphics/TextureLoader.hpp" 

#include <glm/gtc/constants.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "../../../src/cs-utils/utils.hpp"

namespace csp::wfsoverlays {

    const char* PointRenderer::FEATURE_VERT = R"(
    #version 330 core
    layout (location=0) in vec3 aPos;
    layout (location=1) in vec3 aColor;

    uniform mat4    model;
    uniform mat4    view;
    uniform mat4    projection;  

    out vec3 vertexColor;

    out VS_OUT {
        vec3 color;
    } vs_out;

    void main(){
        gl_Position = projection * view * model * vec4(aPos, 1.0);
        vs_out.color = aColor;
    }
    )";

    const char* PointRenderer::FEATURE_GEOM = R"(
    #version 330 core
    layout (points) in;
    layout (triangle_strip, max_vertices = 4) out;

    in VS_OUT {
        vec3 color;
    } gs_in[];

    uniform vec2 circleSize;

    out vec3 vertexColor;
    out vec2 fragTexCoords;

    void build_square(vec4 position) {      // for each point, it will create 4 vertices (which will be later used as texture coordinates)
        
        vertexColor = gs_in[0].color;

        gl_Position =  position + vec4(-circleSize.x, -circleSize.y, 0.1, 0.0);    // 1:bottom-left
        fragTexCoords = vec2(0.0, 0.0);
        EmitVertex();

        gl_Position = position + vec4(circleSize.x, -circleSize.y, 0.1, 0.0);    // 2:bottom-right
        fragTexCoords = vec2(1.0, 0.0);
        EmitVertex();

        gl_Position = position + vec4(-circleSize.x, circleSize.y, 0.1, 0.0);    // 3:top-left
        fragTexCoords = vec2(0.0, 1.0);
        EmitVertex();

        gl_Position = position + vec4(circleSize.x, circleSize.y, 0.1, 0.0);    // 4:top-right
        fragTexCoords = vec2(1.0, 1.0);
        EmitVertex();

        EndPrimitive();
    }

    void main() {
        build_square(gl_in[0].gl_Position);
    }
    )";

    const char* PointRenderer::FEATURE_FRAG = R"(

    in vec3 vertexColor;
    in vec2 fragTexCoords;

    uniform float         uAmbientBrightness;
    uniform float         uSunIlluminance;
    uniform vec3          uSunDirection;
    uniform sampler2D     ourTexture;

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

            FragColor = texture(ourTexture, fragTexCoords) * vec4(result, 1.0);
    }
    )";

    PointRenderer::PointRenderer (std::vector<glm::vec3> coordinates, std::shared_ptr<cs::core::SolarSystem> solarSystem,   
                                        std::shared_ptr<cs::core::Settings> settings, double pointSize, std::shared_ptr<Settings> pluginSettings) {
        
        mSolarSystem    = solarSystem;
        mCoordinates    = coordinates;
        mSettings       = settings;
        mPluginSettings = pluginSettings;
        mPointSizeInput = pointSize;
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

        // Setting the vertex colors attribute pointers
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_TRUE, 6 * sizeof(float), (void*)(3 * sizeof(float)));       
        glEnableVertexAttribArray(1);

        glBindBuffer(GL_ARRAY_BUFFER, 0); 
        glBindVertexArray(0);

        // Recreate the shader if HDR rendering mode are toggled.
        mHDRConnection = mSettings->mGraphics.pEnableHDR.connect([this](bool /*unused*/) { mShaderDirty = true; });

        // load and create a texture
        mTexture = cs::graphics::TextureLoader::loadFromFile("../share/resources/textures/Circle.png");
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    PointRenderer::~PointRenderer() {
    
        mSettings->mGraphics.pEnableHDR.disconnect(mHDRConnection);

        VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
        pSG->GetRoot()->DisconnectChild(mGLNode.get());
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    bool PointRenderer::Do() {

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

            // building and compiling the geometry Shader
            //-------------------------------------------
            unsigned int geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
            glShaderSource(geometryShader, 1, &FEATURE_GEOM, NULL);
            glCompileShader(geometryShader);
            // check for shader compile errors
            glGetShaderiv(geometryShader, GL_COMPILE_STATUS, &success);
            if (!success) {
                glGetShaderInfoLog(geometryShader, 512, NULL, infoLog);
                std::cout << "ERROR::SHADER::GEOMETRY::COMPILATION_FAILED\n" << infoLog << std::endl;
            }

            // linking shaders
            //----------------
            mShaderProgram = glCreateProgram();
            glAttachShader(mShaderProgram, vertexShader);
            glAttachShader(mShaderProgram, geometryShader);
            glAttachShader(mShaderProgram, fragmentShader);
            glLinkProgram(mShaderProgram);
            // check for linking errors
            glGetProgramiv(mShaderProgram, GL_LINK_STATUS, &success);
            if (!success) {
                glGetProgramInfoLog(mShaderProgram, 512, NULL, infoLog);
                std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
            }
            glDeleteShader(vertexShader);
            glDeleteShader(geometryShader);
            glDeleteShader(fragmentShader);
            mShaderDirty = false;
        }

        auto earth = mSolarSystem->getObject("Earth");
        if (!earth || !earth->getIsBodyVisible()) {
            return true;
        }

        glUseProgram(mShaderProgram);
        auto transform = earth->getObserverRelativeTransform();

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

        // Normalizing in terms of the CS windows size in order to have the texture coordinates within the (0,1) interval
        //---------------------------------------------------------------------------------------------------------------
        int width, height;
        VistaViewport* pViewport(GetVistaSystem()->GetDisplayManager()->GetViewports().begin()->second);
        pViewport->GetViewportProperties()->GetSize(width, height);
        float aspectRatio = static_cast<float>(width) / height;
        GLuint circleSize  = glGetUniformLocation(mShaderProgram, "circleSize");
        GLfloat pointSizeGL = static_cast<GLfloat>(mPointSizeInput);
        glUniform2f (circleSize, pointSizeGL, pointSizeGL * aspectRatio);

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

        // Texture settings
        //-----------------
        glPushAttrib(GL_BLEND);
        glPushAttrib(GL_ENABLE_BIT);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        // Texture location and assignment
        GLuint ourTexture = glGetUniformLocation(mShaderProgram, "ourTexture");
        glUniform1i(ourTexture, 0);
        mTexture->Bind(GL_TEXTURE0);

        // Draw
        glBindVertexArray(mVAO); 
        glDisable(GL_CULL_FACE);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(3 * mCoordinates.size()));
        glBindVertexArray(0); 
        glEnable(GL_CULL_FACE);
        mTexture->Unbind(GL_TEXTURE0);
        glPopAttrib();
        glPopAttrib();

        return true;
    }

    bool PointRenderer::GetBoundingBox(VistaBoundingBox& bb) {
        return false;
    }
}
