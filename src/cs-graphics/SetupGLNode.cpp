////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "SetupGLNode.hpp"
#include "logger.hpp"

#include <GL/glew.h>
#include <VistaMath/VistaBoundingBox.h>
#include <array>

namespace cs::graphics {

////////////////////////////////////////////////////////////////////////////////////////////////////

void GLAPIENTRY MessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
    GLsizei length, const GLchar* message, const void* userParams) {

  // get the log level from the settings
  const cs::core::Settings* settings = reinterpret_cast<const cs::core::Settings*>(userParams);

  // Print the following infos (OpenGL errors, shader compile errors, perf. warnings, shader
  // compilation warnings, depricated code, redundant state changes, undefined behaviour, anything
  // that isnt an error or perf. issue)
  if (settings->pLogLevelGL.get() <= spdlog::level::debug &&
      severity == GL_DEBUG_SEVERITY_NOTIFICATION) {
    logger().debug("{}", message);
  }

  // Print the following infos (OpenGL errors, shader compile errors, perf. warnings, shader
  // compilation warnings, depricated code, redundant state changes, undefined behaviour)
  if (settings->pLogLevelGL.get() <= spdlog::level::info && severity == GL_DEBUG_SEVERITY_LOW) {
    logger().info("{}", message);
  }

  // Print the following infos (OpenGL errors, shader compile errors, perf. warnings, shader
  // compilation warnings, depricated code)
  if (settings->pLogLevelGL.get() <= spdlog::level::warn && severity == GL_DEBUG_SEVERITY_MEDIUM) {
    logger().warn("{}", message);
  }

  // Print the following infos (OpenGL errors, shader compile errors)
  if (settings->pLogLevelGL.get() <= spdlog::level::critical &&
      severity == GL_DEBUG_SEVERITY_HIGH) {
    logger().error("{}", message);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

SetupGLNode::SetupGLNode(std::shared_ptr<cs::core::Settings> settings)
    : mSettings(std::move(settings)) {

  mSettings->pLogLevelGL.connectAndTouch([](auto level) {
    if (level == spdlog::level::off) {
      glDisable(GL_DEBUG_OUTPUT);
    } else {
      glEnable(GL_DEBUG_OUTPUT);
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SetupGLNode::Do() {

  // As we are using a reverse projection, we have to change the depth compare mode.
  glDepthFunc(GL_GEQUAL);

  // Also, the winding check needs to be flipped.
  glFrontFace(GL_CW);

  // In CosmoScout VR, we enable face culling per default.
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  // Attach the debug callback to print the messages
  glDebugMessageCallback(MessageCallback, static_cast<void*>(mSettings.get()));

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SetupGLNode::GetBoundingBox(VistaBoundingBox& oBoundingBox) {
  float min(std::numeric_limits<float>::lowest());
  float max(std::numeric_limits<float>::max());

  std::array fMin{min, min, min};
  std::array fMax{max, max, max};

  oBoundingBox.SetBounds(fMin.data(), fMax.data());

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::graphics
