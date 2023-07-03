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

    FeatureRenderer::FeatureRenderer(std::shared_ptr<GeometryBase> feature, std::shared_ptr<cs::core::SolarSystem> solarSystem) {
        mType = feature->mType;
        mSolarSystem = solarSystem;
        auto earth = mSolarSystem->getObject("Earth");
        glm::dvec3 earthRadius = earth->getRadii() * 1.2;

        double phiRadians = 0.0;
        double thetaRadians = 0.0;

        // Depending on the mType parameter, it could be mCoordinates [],
        // mCoordinates [][], mCoordinates [][][], etc

        // TODO: if no data found, display a warning
 
        if (mType == "Point") {
            std::shared_ptr<Point> point = std::dynamic_pointer_cast<Point>(feature);
            phiRadians = (point->mCoordinates[0] * 2.0 * glm::pi<double>()) / 360.0;
            thetaRadians = ((90-point->mCoordinates[1]) * 2.0 * glm::pi<double>())/360.0;

            glm::dvec3 normalizedCoords = { sin(thetaRadians)*sin(phiRadians), cos(thetaRadians), sin(thetaRadians)*cos(phiRadians)};                                   
            glm::vec3 cartesianCoordinates = normalizedCoords * earthRadius;
            coordinates.push_back(cartesianCoordinates); 
        }

        else if (mType == "MultiPoint") {
            std::shared_ptr<MultiPoint> multiPoint = std::dynamic_pointer_cast<MultiPoint>(feature);
            for (int i=0; i < multiPoint->mCoordinates.size(); i++) {
                phiRadians = (multiPoint->mCoordinates[i][0] * 2.0 * glm::pi<double>()) / 360.0;
                thetaRadians = ((90-multiPoint->mCoordinates[i][1]) * 2.0 * glm::pi<double>())/360.0;

                glm::dvec3 normalizedCoords = { sin(thetaRadians)*sin(phiRadians), cos(thetaRadians), sin(thetaRadians)*cos(phiRadians)};                                   
                glm::vec3 cartesianCoordinates = normalizedCoords * earthRadius;
                coordinates.push_back(cartesianCoordinates); 
            };
        }

        else if (mType == "LineString") {
            std::shared_ptr<LineString> lineString = std::dynamic_pointer_cast<LineString>(feature);
            for (int i=0; i < lineString->mCoordinates.size(); i++) {
                phiRadians = (lineString->mCoordinates[i][0] * 2.0 * glm::pi<double>()) / 360.0;
                thetaRadians = ((90-lineString->mCoordinates[i][1]) * 2.0 * glm::pi<double>())/360.0;

                glm::dvec3 normalizedCoords = { sin(thetaRadians)*sin(phiRadians), cos(thetaRadians), sin(thetaRadians)*cos(phiRadians)};                                   
                glm::vec3 cartesianCoordinates = normalizedCoords * earthRadius;
                coordinates.push_back(cartesianCoordinates); 

                if (i != 0 && i != lineString->mCoordinates.size()-1) {
                    coordinates.push_back(cartesianCoordinates);
                }
            };
        }

        else if (mType == "MultiLineString") {
            std::shared_ptr<MultiLineString> multiLineString = std::dynamic_pointer_cast<MultiLineString>(feature);
            for (int i=0; i < multiLineString->mCoordinates.size(); i++) {
                for (int j=0; j < multiLineString->mCoordinates[i].size(); j++) {
                    phiRadians = (multiLineString->mCoordinates[i][j][0] * 2.0 * glm::pi<double>()) / 360.0;
                    thetaRadians = ((90-multiLineString->mCoordinates[i][j][1]) * 2.0 * glm::pi<double>())/360.0;

                    glm::dvec3 normalizedCoords = { sin(thetaRadians)*sin(phiRadians), cos(thetaRadians), sin(thetaRadians)*cos(phiRadians)};                                   
                    glm::vec3 cartesianCoordinates = normalizedCoords * earthRadius;
                    coordinates.push_back(cartesianCoordinates); 
                };
            };
        }

        else if (mType == "Polygon") {
            std::shared_ptr<Polygon> polygon = std::dynamic_pointer_cast<Polygon>(feature);
            for (int i=0; i < polygon->mCoordinates.size(); i++) {
                for (int j=0; j < polygon->mCoordinates[i].size(); j++) {
                    phiRadians = (polygon->mCoordinates[i][j][0] * 2.0 * glm::pi<double>()) / 360.0;
                    thetaRadians = ((90-polygon->mCoordinates[i][j][1]) * 2.0 * glm::pi<double>())/360.0;

                    glm::dvec3 normalizedCoords = { sin(thetaRadians)*sin(phiRadians), cos(thetaRadians), sin(thetaRadians)*cos(phiRadians)};                                   
                    glm::vec3 cartesianCoordinates = normalizedCoords * earthRadius;
                    coordinates.push_back(cartesianCoordinates); 

                    /* if (j != 0 && j != polygon->mCoordinates[i].size()-1) {
                            coordinates.push_back(cartesianCoordinates);            // TODO: Remove the duplicated coordinates?
                    }; */                                                           // TODO: Triangulizar AQUI
                };
            };
        }

        else if (mType == "MultiPolygon") {
            std::shared_ptr<MultiPolygon> multiPolygon = std::dynamic_pointer_cast<MultiPolygon>(feature);
            for (int i=0; i < multiPolygon->mCoordinates.size(); i++) {
                for (int j=0; j < multiPolygon->mCoordinates[i].size(); j++) {
                    for (int k=0; k < multiPolygon->mCoordinates[i][j].size(); k++) {
                        phiRadians = (multiPolygon->mCoordinates[i][j][k][0] * 2.0 * glm::pi<double>()) / 360.0;
                        thetaRadians = ((90-multiPolygon->mCoordinates[i][j][k][1]) * 2.0 * glm::pi<double>())/360.0;

                        glm::dvec3 normalizedCoords = { sin(thetaRadians)*sin(phiRadians), cos(thetaRadians), sin(thetaRadians)*cos(phiRadians)};                                   
                        glm::vec3 cartesianCoordinates = normalizedCoords * earthRadius;
                        coordinates.push_back(cartesianCoordinates); 

                        /* if (k != 0 && k != multiPolygon->mCoordinates[i][j].size()-1) {
                            coordinates.push_back(cartesianCoordinates);                // TODO: Remove the duplicated coordinates?
                        }; */                                                           // TODO: Triangulizar AQUI
                    };
                };
            };
        }

        else { 
            logger().warn(" {} data could not be rendered", mType); 
        }

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

        // Generating the VBO to manage the memory of the input vertex
        unsigned int VBO;                                                                   
        glGenBuffers(1, &VBO);

        // Binding the VAO
        glBindVertexArray(VAO);                                                             
        // Binding the VBO
        glBindBuffer(GL_ARRAY_BUFFER, VBO); 

        // Copying user-defined data into the current bound buffer                                                                                                        
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3 * coordinates.size(), coordinates.data(), GL_STATIC_DRAW);   // TODO: Here is where the coordinates are       

        // Setting the vertex attributes pointers
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);       
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

        GLenum drawingType;

        if (mType == "Point" || mType =="MultiPoint") {
            drawingType=GL_POINTS;
        }

        else if (mType == "LineString" || mType =="MultiLineString") {
            drawingType=GL_LINES;
        }

        else if (mType == "Polygon" || mType =="MultiPolygon") {
            drawingType=GL_LINES;   // TODO: Later we'll have to set it back to polygon
        }

        // Draw
        glBindVertexArray(VAO); 
        glPointSize(5.0);
        glLineWidth(5.0);

        glDrawArrays(drawingType, 0, static_cast<GLsizei>(3 * coordinates.size()));
        glBindVertexArray(0); 
        return true;
    }

    bool FeatureRenderer::GetBoundingBox(VistaBoundingBox& bb) {
        return false;
    }
}


/*
void Rasterizer::triangle(const Vertex &v1, const Vertex &v2, const Vertex &v3)
{
    // 28.4 fixed-point coordinates
    const int Y1 = iround(16.0f * v1.y);
    const int Y2 = iround(16.0f * v2.y);
    const int Y3 = iround(16.0f * v3.y);

    const int X1 = iround(16.0f * v1.x);
    const int X2 = iround(16.0f * v2.x);
    const int X3 = iround(16.0f * v3.x);

    // Deltas
    const int DX12 = X1 - X2;
    const int DX23 = X2 - X3;
    const int DX31 = X3 - X1;

    const int DY12 = Y1 - Y2;
    const int DY23 = Y2 - Y3;
    const int DY31 = Y3 - Y1;

    // Fixed-point deltas
    const int FDX12 = DX12 << 4;
    const int FDX23 = DX23 << 4;
    const int FDX31 = DX31 << 4;

    const int FDY12 = DY12 << 4;
    const int FDY23 = DY23 << 4;
    const int FDY31 = DY31 << 4;

    // Bounding rectangle
    int minx = (min(X1, X2, X3) + 0xF) >> 4;
    int maxx = (max(X1, X2, X3) + 0xF) >> 4;
    int miny = (min(Y1, Y2, Y3) + 0xF) >> 4;
    int maxy = (max(Y1, Y2, Y3) + 0xF) >> 4;

    // Block size, standard 8x8 (must be power of two)
    const int q = 8;

    // Start in corner of 8x8 block
    minx &= ~(q - 1);
    miny &= ~(q - 1);

    (char*&)colorBuffer += miny * stride;

    // Half-edge constants
    int C1 = DY12 * X1 - DX12 * Y1;
    int C2 = DY23 * X2 - DX23 * Y2;
    int C3 = DY31 * X3 - DX31 * Y3;

    // Correct for fill convention
    if(DY12 < 0 || (DY12 == 0 && DX12 > 0)) C1++;
    if(DY23 < 0 || (DY23 == 0 && DX23 > 0)) C2++;
    if(DY31 < 0 || (DY31 == 0 && DX31 > 0)) C3++;

    // Loop through blocks
    for(int y = miny; y < maxy; y += q)
    {
        for(int x = minx; x < maxx; x += q)
        {
            // Corners of block
            int x0 = x << 4;
            int x1 = (x + q - 1) << 4;
            int y0 = y << 4;
            int y1 = (y + q - 1) << 4;

            // Evaluate half-space functions
            bool a00 = C1 + DX12 * y0 - DY12 * x0 > 0;
            bool a10 = C1 + DX12 * y0 - DY12 * x1 > 0;
            bool a01 = C1 + DX12 * y1 - DY12 * x0 > 0;
            bool a11 = C1 + DX12 * y1 - DY12 * x1 > 0;
            int a = (a00 << 0) | (a10 << 1) | (a01 << 2) | (a11 << 3);

            bool b00 = C2 + DX23 * y0 - DY23 * x0 > 0;
            bool b10 = C2 + DX23 * y0 - DY23 * x1 > 0;
            bool b01 = C2 + DX23 * y1 - DY23 * x0 > 0;
            bool b11 = C2 + DX23 * y1 - DY23 * x1 > 0;
            int b = (b00 << 0) | (b10 << 1) | (b01 << 2) | (b11 << 3);

            bool c00 = C3 + DX31 * y0 - DY31 * x0 > 0;
            bool c10 = C3 + DX31 * y0 - DY31 * x1 > 0;
            bool c01 = C3 + DX31 * y1 - DY31 * x0 > 0;
            bool c11 = C3 + DX31 * y1 - DY31 * x1 > 0;
            int c = (c00 << 0) | (c10 << 1) | (c01 << 2) | (c11 << 3);

            // Skip block when outside an edge
            if(a == 0x0 || b == 0x0 || c == 0x0) continue;

            unsigned int *buffer = colorBuffer;

            // Accept whole block when totally covered
            if(a == 0xF && b == 0xF && c == 0xF)
            {
                for(int iy = 0; iy < q; iy++)
                {
                    for(int ix = x; ix < x + q; ix++)
                    {
                        buffer[ix] = 0x00007F00;<< // Green
                    }

                    (char*&)buffer += stride;
                }
            }
            else<< // Partially covered block
            {
                int CY1 = C1 + DX12 * y0 - DY12 * x0;
                int CY2 = C2 + DX23 * y0 - DY23 * x0;
                int CY3 = C3 + DX31 * y0 - DY31 * x0;

                for(int iy = y; iy < y + q; iy++)
                {
                    int CX1 = CY1;
                    int CX2 = CY2;
                    int CX3 = CY3;

                    for(int ix = x; ix < x + q; ix++)
                    {
                        if(CX1 > 0 && CX2 > 0 && CX3 > 0)
                        {
                            buffer[ix] = 0x0000007F;<< // Blue
                        }

                        CX1 -= FDY12;
                        CX2 -= FDY23;
                        CX3 -= FDY31;
                    }

                    CY1 += FDX12;
                    CY2 += FDX23;
                    CY3 += FDX31;

                    (char*&)buffer += stride;
                }
            }
        }

        (char*&)colorBuffer += q * stride;
    }
}




*/