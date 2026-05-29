////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SolarFlares.hpp"

#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-graphics/TextureLoader.hpp"
#include "../../../src/cs-utils/FrameStats.hpp"
#include "../../../src/cs-core/TimeControl.hpp"
#include "logger.hpp"

#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernel/GraphicsManager/VistaGraphicsManager.h>
#include <VistaKernel/GraphicsManager/VistaGroupNode.h>

#include <glm/gtc/type_ptr.hpp>

namespace csp::visualeffects {

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* SOLAR_FLARES_SHADER_VERT = R"(
#version 330

// inputs
layout(location = 0) in vec3 inPosition;  // Position of quad from (-1, -1, 0) to (1, 1, 0)

// outputs
out vec2 vTexCoord;

// uniforms
uniform mat4 uMatModelView;
uniform mat4 uMatProjection;

void main()
{

    vTexCoord = vec2(inPosition.x + 1.0, inPosition.y + 1.0) * 0.5;
    vec4 pos = uMatModelView * vec4(inPosition.xyz, 1);
    gl_Position = uMatProjection * pos;
})";

////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* SOLAR_FLARES_SHADER_FRAG = R"(
#version 330

// inputs 
in vec2 vTexCoord;

// outputs
layout(location = 0) out vec4 vOutColor;

// uniforms
uniform vec2      uResolution; 
uniform float     uTime;
uniform sampler2D uNoiseTexture;

// Function originally from shadertoy: https://www.shadertoy.com/view/4sycRW
void mainImage(out vec4 O,vec2 I)
{
  //Centered coordinates.
  vec2 r = uResolution.xy,
  p = (I+I-r)*2.0/r.y;
  
  //Initialize floats.
  float t=uTime, d=dot(p,p), i = 1., l;
  //Iterate through radii.
  O = vec4(0.0);
  for(float i = 1.0; i < 1.7; i += 0.01)
  {
    //Calculate rotating rays.
    vec3 s = vec3(p*mat2(cos(i*.1+vec4(0.0,11.0,33.0,0.0))), sqrt(i-d));
    //Sample texture with filter.
    O += pow(texture(uNoiseTexture,s.xy/sqrt(s.z)-.1*t),i*4.-vec4(3));
  }
    
  //Create edge glow and attenuation in space.
  // O *= vec4(65.0,40.0,25.0,1.0)/4000.0/(abs(d-1.0)+0.2);
  O *= 1.0 / 100.0 / (abs(d-1.0)+0.2);
  //Calculate disk distance for solar flares.
  l = 1.5-length(p+p.y*.4+.1*cos(p.x*6.+.2*t));
  
  //Added center glow.
  O += vec4(2,1,.5,0)*( exp(-d) +
//Calculate the solar flares (math magic).
  .1*smoothstep(.8, 1., cos(t/8.+p.x+p.y*.4)) *
  (cos(p.y*8.+t)*.3+.7) * exp(cos(p.x+p.y*.4+t)
  -abs((cos(l*38.+p.x*17.+t+t)*.1+.9)*l)/.1) );
  
}

void main()
{
  vec2 fragCoord = vTexCoord * uResolution.xy;

  vec4 color = vec4(0.0);
  mainImage(color, fragCoord);

  //float modulo = mod(uTime, 1.0);
  //color = vec4(modulo, 1.0, 1.0, 1.0);

  vOutColor = color;
})";

////////////////////////////////////////////////////////////////////////////////////////////////////

SolarFlares::SolarFlares(std::shared_ptr<Plugin::Settings>  pluginSettings,
    std::shared_ptr<cs::core::SolarSystem>      solarSystem,
    std::shared_ptr<cs::core::TimeControl> timeControl
  ) :
      mPluginSettings(std::move(pluginSettings)),
      mSolarSystem(std::move(solarSystem)),
      mTimeControl(std::move(timeControl))
  {

    // Add to scenegraph.
    VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
    mGLNode.reset(pSG->NewOpenGLNode(pSG->GetRoot(), this));
    VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
        mGLNode.get(), static_cast<int>(cs::utils::DrawOrder::eTransparentItems) - 1);
    logger().info("Added SolarFlares to scene graph.");

    // Solar flare will be depicted on a simple quad.
    std::vector<float> quadVertices = {
        // First Triangle
        -1.0f, -1.0f,  0.0f, // Bottom Left
         1.0f, -1.0f,  0.0f, // Bottom Right
        -1.0f,  1.0f,  0.0f, // Top Left

        // Second Triangle
         1.0f, -1.0f,  0.0f, // Bottom Right
         1.0f,  1.0f,  0.0f, // Top Right
        -1.0f,  1.0f,  0.0f  // Top Left
    };

    // Remember vertex count of quad for drawing.
    mVertexCount = 6;

    mNoiseTexture = cs::graphics::TextureLoader::loadFromFile("../share/resources/textures/sun_noise.jpg");

    // Create VBO and VAO from given vertices.
    mVBO = std::make_unique<VistaBufferObject>();
    mVAO = std::make_unique<VistaVertexArrayObject>();

    mVAO->Bind();

    mVBO->Bind(GL_ARRAY_BUFFER);
    mVBO->BufferData(quadVertices.size() * sizeof(float), quadVertices.data(), GL_DYNAMIC_DRAW);

    mVAO->EnableAttributeArray(0);
    mVAO->SpecifyAttributeArrayFloat(0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, 0, mVBO.get());

    mVAO->Release();
    mVBO->Release();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SolarFlares::~SolarFlares() {
    VistaSceneGraph* pSG = GetVistaSystem()->GetGraphicsManager()->GetSceneGraph();
    pSG->GetRoot()->DisconnectChild(mGLNode.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarFlares::setParentName(std::string objectName) {
  mParentName = std::move(objectName);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& SolarFlares::getParentName() const {
  return mParentName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarFlares::update(double tTime) {
  return;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SolarFlares::Do() {
  auto parent = mSolarSystem->getObject(mParentName);

  if ((!parent || !parent->getIsBodyVisible())) {
    return true;
  }

  if (!mPlayBackTimeSet) {
    mPlaybackStartTime = static_cast<float>(mTimeControl->pSimulationTime.get());
    logger().info("Set solar flare playback start time to {}.", mPlaybackStartTime);
    mPlayBackTimeSet = true;
  }

  // Create shader
  createShader();

  // Get observer relative transform and extract the upper left 3x3 matrix.
  auto matMV = parent->getObserverRelativeTransform();

  // Remember the original position of the parent object.
  glm::dvec3 position = glm::dvec3(matMV[3]);

  // Remember the original scaling of the parent object.
  double scaleX = glm::length(glm::dvec3(matMV[0]));
  double scaleY = glm::length(glm::dvec3(matMV[1]));
  double scaleZ = glm::length(glm::dvec3(matMV[2]));

  // Set matrix to identity.
  matMV = glm::dmat4(1.0);

  // Size of the panel.
  double size = 1000.0 * 1000.0 * 1000.0 * 1.2; // TODO: Currently hardcoded size of panel.
  
  // Inject the original scaling together with the desired size of the panel.
  matMV[0][0] = scaleX * size;
  matMV[1][1] = scaleY * size;
  matMV[2][2] = scaleZ * size;

  // Inject the original position.
  matMV[3] = glm::dvec4(position, 1.0);

  // Get projection matrix.
  std::array<GLfloat, 16> glMatP{};
  glGetFloatv(GL_PROJECTION_MATRIX, glMatP.data());

  glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_LINE_BIT);
  glDisable(GL_CULL_FACE);
  //glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE);

  mShader->Bind();

  // Set uniforms
  glUniformMatrix4fv(mUniforms.modelViewMatrix, 1, GL_FALSE, glm::value_ptr(glm::highp_mat4(matMV)));
  glUniformMatrix4fv(mUniforms.projectionMatrix, 1, GL_FALSE, glMatP.data());

  // CONTINUE HERE SET UNIFORMS
  mShader->SetUniform(mUniforms.time, static_cast<float>(mTimeControl->pSimulationTime.get() - mPlaybackStartTime));
  mShader->SetUniform(mUniforms.resolution, 256.0f, 256.0f);

  mShader->SetUniform(mUniforms.noiseTexture, 0);
  mNoiseTexture->Bind(GL_TEXTURE0);

  mVAO->Bind();

  // Draw panel with solar flare vfx.
  glDrawArrays(GL_TRIANGLES, 0, mVertexCount);

  // Cleanup
  glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
    
  mShader->Release();
  mVAO->Release();

  mNoiseTexture->Unbind(GL_TEXTURE0);

  glPopAttrib();
  
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SolarFlares::createShader() {
  mShader = std::make_unique<VistaGLSLShader>();

  std::string sVert(SOLAR_FLARES_SHADER_VERT);
  std::string sFrag(SOLAR_FLARES_SHADER_FRAG);

  mShader->InitVertexShaderFromString(sVert);
  mShader->InitFragmentShaderFromString(sFrag);
  mShader->Link();

  mUniforms.time  = mShader->GetUniformLocation("uTime");
  mUniforms.resolution  = mShader->GetUniformLocation("uResolution");
  mUniforms.noiseTexture  = mShader->GetUniformLocation("uNoiseTexture");
  mUniforms.modelViewMatrix  = mShader->GetUniformLocation("uMatModelView");
  mUniforms.projectionMatrix = mShader->GetUniformLocation("uMatProjection");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SolarFlares::GetBoundingBox(VistaBoundingBox& /*bb*/) {
  return false;
}

} // namespace csp::visualeffects