////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Stars.hpp"

#include "logger.hpp"

#include "../../../src/cs-graphics/TextureLoader.hpp"
#include "../../../src/cs-utils/FrameStats.hpp"
#include "../../../src/cs-utils/filesystem.hpp"

#ifdef _WIN32
#include <Windows.h>
#endif

#include <VistaInterProcComm/Connections/VistaByteBufferDeSerializer.h>
#include <VistaInterProcComm/Connections/VistaByteBufferSerializer.h>
#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/GraphicsManager/VistaGeometryFactory.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/GraphicsManager/VistaSceneGraph.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaOGLUtils.h>
#include <VistaOGLExt/VistaTexture.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>
#include <VistaTools/tinyXML/tinyxml.h>

#include <array>
#include <fstream>
#include <glm/glm.hpp>

namespace csp::stars {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool fromString(std::string const& v, T& out) {
  std::istringstream iss(v);
  iss >> out;
  return (iss.rdstate() & std::stringstream::failbit) == 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::array<std::array<int, Stars::NUM_COLUMNS>, Stars::NUM_CATALOGS> Stars::cColumnMapping{
    std::array{34, 11, 8, 9, 31}, // CatalogType::eHipparcos
    std::array{34, 11, 8, 9, 31}, // CatalogType::eTycho
    std::array{19, -1, 2, 3, 23}, // CatalogType::eTycho2
    std::array{5, 4, 2, 3, 1}     // CatalogType::eGaia
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Increase this if the cache format changed and is incompatible now. This will
// force a reload.
const int Stars::cCacheVersion = 4;

////////////////////////////////////////////////////////////////////////////////////////////////////

Stars::Stars() {
  for (auto const& viewport : GetVistaSystem()->GetDisplayManager()->GetViewports()) {
    mSRTargets[viewport.second] = {};
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stars::setCatalogs(std::map<Stars::CatalogType, std::string> catalogs) {
  if (mCatalogs != catalogs) {

    mCatalogs = std::move(catalogs);

    // Clear stars first.
    mStars.clear();

    // Read star catalogs.
    if (!readStarCache(mCacheFile)) {
      std::map<CatalogType, std::string>::const_iterator it;

      it = mCatalogs.find(CatalogType::eHipparcos);
      if (it != mCatalogs.end()) {
        readStarsFromCatalog(it->first, it->second);
      }

      it = mCatalogs.find(CatalogType::eTycho);
      if (it != mCatalogs.end()) {
        readStarsFromCatalog(it->first, it->second);
      }

      it = mCatalogs.find(CatalogType::eTycho2);
      if (it != mCatalogs.end()) {
        // Do not load tycho and tycho 2.
        if (mCatalogs.find(CatalogType::eTycho) == mCatalogs.end()) {
          readStarsFromCatalog(it->first, it->second);
        } else {
          logger().warn("Failed to load Tycho2 catalog: Tycho already loaded!");
        }
      }

      it = mCatalogs.find(CatalogType::eGaia);
      if (it != mCatalogs.end()) {
        // Do not load gaia together with tycho or tycho 2.
        if (mCatalogs.find(CatalogType::eTycho) == mCatalogs.end() &&
            mCatalogs.find(CatalogType::eTycho2) == mCatalogs.end()) {
          readStarsFromCatalog(it->first, it->second);
        } else {
          logger().warn("Failed to load Gaia catalog: Tycho already loaded!");
        }
      }

      if (!mStars.empty()) {
        writeStarCache(mCacheFile);
      } else {
        logger().warn("Loaded no stars! Stars will not work properly.");
      }
    }

    // Create buffers,
    buildStarVAO();
    buildBackgroundVAO();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::map<Stars::CatalogType, std::string> const& Stars::getCatalogs() const {
  return mCatalogs;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stars::setCacheFile(std::string cacheFile) {
  mCacheFile = std::move(cacheFile);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& Stars::getCacheFile() const {
  return mCacheFile;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stars::setDrawMode(Stars::DrawMode value) {
  if (mDrawMode != value) {
    mShaderDirty = true;
    mDrawMode    = value;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Stars::DrawMode Stars::getDrawMode() const {
  return mDrawMode;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stars::setEnableHDR(bool value) {
  if (mEnableHDR != value) {
    mShaderDirty = true;
    mEnableHDR   = value;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stars::setSolidAngle(float value) {
  mSolidAngle = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float Stars::getSolidAngle() const {
  return mSolidAngle;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Stars::getEnableHDR() const {
  return mEnableHDR;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stars::setMinMagnitude(float value) {
  mMinMagnitude = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float Stars::getMinMagnitude() const {
  return mMinMagnitude;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stars::setMaxMagnitude(float value) {
  mMaxMagnitude = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float Stars::getMaxMagnitude() const {
  return mMaxMagnitude;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stars::setApproximateSceneBrigthness(float value) {
  mApproximateSceneBrightness = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float Stars::getApproximateSceneBrigthness() const {
  return mApproximateSceneBrightness;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stars::setLuminanceMultiplicator(float value) {
  mLuminanceMultiplicator = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float Stars::getLuminanceMultiplicator() const {
  return mLuminanceMultiplicator;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stars::setCelestialGridColor(const VistaColor& value) {
  mBackgroundColor1 = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaColor const& Stars::getCelestialGridColor() const {
  return mBackgroundColor1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stars::setStarFiguresColor(VistaColor const& value) {
  mBackgroundColor2 = value;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaColor const& Stars::getStarFiguresColor() const {
  return mBackgroundColor2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stars::setStarTexture(std::string const& filename) {
  if (filename != mStarTextureFile) {
    mStarTextureFile = filename;
    if (filename.empty()) {
      mStarTexture.reset();
    } else {
      mStarTexture = cs::graphics::TextureLoader::loadFromFile(filename);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stars::setCelestialGridTexture(std::string const& filename) {
  if (filename != mCelestialGridTextureFile) {
    mCelestialGridTextureFile = filename;
    if (filename.empty()) {
      mCelestialGridTexture.reset();
    } else {
      mCelestialGridTexture = cs::graphics::TextureLoader::loadFromFile(filename);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stars::setStarFiguresTexture(std::string const& filename) {
  if (filename != mStarFiguresTextureFile) {
    mStarFiguresTextureFile = filename;
    if (filename.empty()) {
      mStarFiguresTexture.reset();
    } else {
      mStarFiguresTexture = cs::graphics::TextureLoader::loadFromFile(filename);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Stars::Do() {
  // Add a lower bound to the scene brightness value so that we can show the stars with the
  // mLuminanceMultiplicator even if we are in full daylight.
  float sceneBrightness = (1.F - mApproximateSceneBrightness) + 0.001F;

  // Stars are not visible with a low luminance multiplicator
  if (mLuminanceMultiplicator * sceneBrightness < 0.005F) {
    return true;
  }

  cs::utils::FrameStats::ScopedTimer             timer("Render Stars");
  cs::utils::FrameStats::ScopedSamplesCounter    samplesCounter("Render Stars");
  cs::utils::FrameStats::ScopedPrimitivesCounter primitivesCounter("Render Stars");

  // Save current state of the OpenGL state machine.
  glPushAttrib(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT);
  glDepthMask(GL_FALSE);
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE);

  // Get matrices.
  std::array<GLfloat, 16> glMat{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMat.data());
  VistaTransformMatrix matModelView(glMat.data(), true);

  glGetFloatv(GL_PROJECTION_MATRIX, glMat.data());
  VistaTransformMatrix matProjection(glMat.data(), true);

  if (mShaderDirty) {
    std::string defines;

    // The compute shader needs a higher version for gl_GlobalInvocationID.
    if (mDrawMode == DrawMode::eSRPoint) {
      defines += "#version 430\n";
    } else {
      defines += "#version 330\n";
    }

    if (mEnableHDR) {
      defines += "#define ENABLE_HDR\n";
    }

    if (mDrawMode == DrawMode::eSmoothPoint) {
      defines += "#define DRAWMODE_SMOOTH_POINT\n";
    } else if (mDrawMode == DrawMode::ePoint) {
      defines += "#define DRAWMODE_POINT\n";
    } else if (mDrawMode == DrawMode::eDisc) {
      defines += "#define DRAWMODE_DISC\n";
    } else if (mDrawMode == DrawMode::eSmoothDisc) {
      defines += "#define DRAWMODE_SMOOTH_DISC\n";
    } else if (mDrawMode == DrawMode::eScaledDisc) {
      defines += "#define DRAWMODE_SCALED_DISC\n";
    } else if (mDrawMode == DrawMode::eGlareDisc) {
      defines += "#define DRAWMODE_GLARE_DISC\n";
    } else if (mDrawMode == DrawMode::eSprite) {
      defines += "#define DRAWMODE_SPRITE\n";
    } else if (mDrawMode == DrawMode::eSRPoint) {
      defines += "#define DRAWMODE_SRPOINT\n";
    }

    defines += cs::utils::filesystem::loadToString("../share/resources/shaders/starSnippets.glsl");

    mStarShader = VistaGLSLShader();
    if (mDrawMode == DrawMode::ePoint || mDrawMode == DrawMode::eSmoothPoint) {
      mStarShader.InitVertexShaderFromString(
          defines +
          cs::utils::filesystem::loadToString("../share/resources/shaders/starsOnePixel.vert"));
      mStarShader.InitFragmentShaderFromString(
          defines +
          cs::utils::filesystem::loadToString("../share/resources/shaders/starsOnePixel.frag"));
    } else if (mDrawMode == DrawMode::eSRPoint) {
      mStarShader.InitComputeShaderFromString(
          defines +
          cs::utils::filesystem::loadToString("../share/resources/shaders/starsSRPoint.comp"));

      mSRBlitShader = VistaGLSLShader();
      mSRBlitShader.InitVertexShaderFromString(
          defines +
          cs::utils::filesystem::loadToString("../share/resources/shaders/starsSRBlit.vert"));
      mSRBlitShader.InitFragmentShaderFromString(
          defines +
          cs::utils::filesystem::loadToString("../share/resources/shaders/starsSRBlit.frag"));
      mSRBlitShader.Link();
    } else {
      mStarShader.InitVertexShaderFromString(
          defines +
          cs::utils::filesystem::loadToString("../share/resources/shaders/starsBillboard.vert"));
      mStarShader.InitGeometryShaderFromString(
          defines +
          cs::utils::filesystem::loadToString("../share/resources/shaders/starsBillboard.geom"));
      mStarShader.InitFragmentShaderFromString(
          defines +
          cs::utils::filesystem::loadToString("../share/resources/shaders/starsBillboard.frag"));
    }

    mStarShader.Link();

    mBackgroundShader = VistaGLSLShader();
    mBackgroundShader.InitVertexShaderFromString(
        defines +
        cs::utils::filesystem::loadToString("../share/resources/shaders/starsBackground.vert"));
    mBackgroundShader.InitFragmentShaderFromString(
        defines +
        cs::utils::filesystem::loadToString("../share/resources/shaders/starsBackground.frag"));
    mBackgroundShader.Link();

    mUniforms.bgInverseMVMatrix  = mBackgroundShader.GetUniformLocation("uInvMV");
    mUniforms.bgInverseMVPMatrix = mBackgroundShader.GetUniformLocation("uInvMVP");
    mUniforms.bgTexture          = mBackgroundShader.GetUniformLocation("iTexture");
    mUniforms.bgColor            = mBackgroundShader.GetUniformLocation("cColor");

    mUniforms.starResolution   = mStarShader.GetUniformLocation("uResolution");
    mUniforms.starTexture      = mStarShader.GetUniformLocation("uStarTexture");
    mUniforms.starMinMagnitude = mStarShader.GetUniformLocation("uMinMagnitude");
    mUniforms.starMaxMagnitude = mStarShader.GetUniformLocation("uMaxMagnitude");
    mUniforms.starSolidAngle   = mStarShader.GetUniformLocation("uSolidAngle");

    if (mDrawMode == DrawMode::eSRPoint) {
      mUniforms.starLuminanceMul = mSRBlitShader.GetUniformLocation("uLuminanceMultiplicator");
    } else {
      mUniforms.starLuminanceMul = mStarShader.GetUniformLocation("uLuminanceMultiplicator");
    }

    mUniforms.starMVMatrix        = mStarShader.GetUniformLocation("uMatMV");
    mUniforms.starPMatrix         = mStarShader.GetUniformLocation("uMatP");
    mUniforms.starInverseMVMatrix = mStarShader.GetUniformLocation("uInvMV");
    mUniforms.starInversePMatrix  = mStarShader.GetUniformLocation("uInvP");

    if (mDrawMode == DrawMode::eSRPoint) {
      mUniforms.starCount = mStarShader.GetUniformLocation("uStarCount");
    }

    mShaderDirty = false;
  }

  // Draw background images.
  if ((mCelestialGridTexture && mBackgroundColor1[3] != 0.F) ||
      (mStarFiguresTexture && mBackgroundColor2[3] != 0.F)) {
    mBackgroundVAO.Bind();
    mBackgroundShader.Bind();
    mBackgroundShader.SetUniform(mUniforms.bgTexture, 0);

    float fadeOut =
        mEnableHDR ? 0.001F * mLuminanceMultiplicator : mLuminanceMultiplicator * sceneBrightness;

    VistaTransformMatrix matMVNoTranslation = matModelView;

    // Reduce jitter.
    matMVNoTranslation[0][3] = 0.F;
    matMVNoTranslation[1][3] = 0.F;
    matMVNoTranslation[2][3] = 0.F;

    VistaTransformMatrix matMVP(matProjection * matMVNoTranslation);
    VistaTransformMatrix matInverseMVP(matMVP.GetInverted());
    VistaTransformMatrix matInverseMV(matMVNoTranslation.GetInverted());

    glUniformMatrix4fv(mUniforms.bgInverseMVPMatrix, 1, GL_FALSE, matInverseMVP.GetData());
    glUniformMatrix4fv(mUniforms.bgInverseMVMatrix, 1, GL_FALSE, matInverseMV.GetData());

    if (mCelestialGridTexture && mBackgroundColor1[3] != 0.F) {
      mBackgroundShader.SetUniform(mUniforms.bgColor, mBackgroundColor1[0], mBackgroundColor1[1],
          mBackgroundColor1[2], mBackgroundColor1[3] * fadeOut);
      mCelestialGridTexture->Bind(GL_TEXTURE0);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      mCelestialGridTexture->Unbind(GL_TEXTURE0);
    }

    if (mStarFiguresTexture && mBackgroundColor2[3] != 0.F) {
      mBackgroundShader.SetUniform(mUniforms.bgColor, mBackgroundColor2[0], mBackgroundColor2[1],
          mBackgroundColor2[2], mBackgroundColor2[3] * fadeOut);
      mStarFiguresTexture->Bind(GL_TEXTURE0);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      mStarFiguresTexture->Unbind(GL_TEXTURE0);
    }

    mBackgroundShader.Release();
    mBackgroundVAO.Release();
  }

  // Draw stars. In software rasterization mode, we need to bind the VBO as SSBO.
  if (mDrawMode == DrawMode::eSRPoint) {
    mStarVBO.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0);
  } else {
    mStarVAO.Bind();
  }

  mStarShader.Bind();

  if (mDrawMode == DrawMode::ePoint || mDrawMode == DrawMode::eSmoothPoint) {
    glPointSize(0.5F);
  }

  if (mDrawMode == DrawMode::eSmoothPoint) {
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
  } else {
    glDisable(GL_POINT_SMOOTH);
  }

  std::array<int, 4> viewport{};
  glGetIntegerv(GL_VIEWPORT, viewport.data());

  mStarShader.SetUniform(mUniforms.starResolution, static_cast<float>(viewport.at(2)),
      static_cast<float>(viewport.at(3)));

  if (mDrawMode == DrawMode::eSprite) {
    mStarTexture->Bind(GL_TEXTURE0);
    mStarShader.SetUniform(mUniforms.starTexture, 0);
  }

  mStarShader.SetUniform(mUniforms.starMinMagnitude, mMinMagnitude);
  mStarShader.SetUniform(mUniforms.starMaxMagnitude, mMaxMagnitude);
  mStarShader.SetUniform(mUniforms.starSolidAngle, mSolidAngle);

  float fadeOut = mEnableHDR ? 1.F : sceneBrightness;
  if (mDrawMode != DrawMode::eSRPoint) {
    mStarShader.SetUniform(mUniforms.starLuminanceMul, mLuminanceMultiplicator * fadeOut);
  }

  VistaTransformMatrix matInverseMV(matModelView.GetInverted());
  VistaTransformMatrix matInverseP(matProjection.GetInverted());

  glUniformMatrix4fv(mUniforms.starMVMatrix, 1, GL_FALSE, matModelView.GetData());
  glUniformMatrix4fv(mUniforms.starPMatrix, 1, GL_FALSE, matProjection.GetData());
  glUniformMatrix4fv(mUniforms.starInverseMVMatrix, 1, GL_FALSE, matInverseMV.GetData());
  glUniformMatrix4fv(mUniforms.starInversePMatrix, 1, GL_FALSE, matInverseP.GetData());

  // The software rasterization mode is implemented as a compute shader and a blit shader.
  // The compute pass accumulates the star luminance and color temperatures in a 2D texture. The
  // blit shader then reads the texture, computes the final color for each pixel and writes it to
  // the framebuffer.
  if (mDrawMode == DrawMode::eSRPoint) {
    // Recreate the render targets if the viewport size changed.
    auto* viewport = GetVistaSystem()->GetDisplayManager()->GetCurrentRenderInfo()->m_pViewport;
    auto& data     = mSRTargets[viewport];

    int width, height;
    viewport->GetViewportProperties()->GetSize(width, height);

    if (data.mWidth != width || data.mHeight != height) {
      data.mWidth  = width;
      data.mHeight = height;

      data.mImage = std::make_unique<VistaTexture>(GL_TEXTURE_2D);
      data.mImage->Bind();
      data.mImage->SetWrapS(GL_CLAMP_TO_EDGE);
      data.mImage->SetWrapT(GL_CLAMP_TO_EDGE);
      data.mImage->SetMinFilter(GL_NEAREST);
      data.mImage->SetMagFilter(GL_NEAREST);
      glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, width, height);
    }

    glUniform1i(mUniforms.starCount, static_cast<int>(mStars.size()));

    {
      cs::utils::FrameStats::ScopedTimer timer("Software Rasterizer");
      glClearTexImage(data.mImage->GetId(), 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);
      glBindImageTexture(0, data.mImage->GetId(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI);

      glDispatchCompute(static_cast<uint32_t>(std::ceil(1.0 * mStars.size() / 256)), 1, 1);
      glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    }

    {
      cs::utils::FrameStats::ScopedTimer timer("Blit Results");
      mSRBlitShader.Bind();
      mSRBlitShader.SetUniform(mUniforms.starLuminanceMul, mLuminanceMultiplicator * fadeOut);

      data.mImage->Bind(GL_TEXTURE0);

      glDrawArrays(GL_TRIANGLES, 0, 3);
    }

  } else {
    // The other draw modes are very simple. They are either using point primitives or a geometry
    // shader to create the star billboards.
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(mStars.size()));
    mStarVAO.Release();
  }

  if (mDrawMode == DrawMode::eSprite) {
    mStarTexture->Unbind(GL_TEXTURE0);
  }

  mStarShader.Release();

  glDepthMask(GL_TRUE);
  glPopAttrib();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Stars::GetBoundingBox(VistaBoundingBox& oBoundingBox) {
  float      min(std::numeric_limits<float>::min());
  float      max(std::numeric_limits<float>::max());
  std::array fMin{min, min, min};
  std::array fMax{max, max, max};

  oBoundingBox.SetBounds(fMin.data(), fMax.data());

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Stars::readStarsFromCatalog(CatalogType type, std::string const& filename) {
  bool success = false;
  logger().info("Reading star catalog '{}'.", filename);

  std::ifstream file;

  try {
    file.open(filename.c_str(), std::ifstream::in);
  } catch (std::exception& e) {
    logger().error("Failed to open catalog file '{}': {}", filename, e.what());
  }

  if (file.is_open()) {
    int  lineCount = 0;
    bool loadHipparcos(mCatalogs.find(CatalogType::eHipparcos) != mCatalogs.end());

    // read line by line
    while (!file.eof()) {
      // get line
      ++lineCount;
      std::string line;
      getline(file, line);

      // parse line:
      // separate complete items consisting of "val0|val1|...|valN|" into vector of value strings
      std::vector<std::string> items = cs::utils::splitString(line, '|');

      // convert value strings to int/double/float and save in star data structure
      // expecting Hipparcos or Tycho-1 catalog and more than 5 columns
      if (items.size() > 5) {
        // skip if part of hipparcos catalogue
        if (type != CatalogType::eHipparcos && loadHipparcos) {
          int  hippID{};
          bool read = fromString<int>(items[cColumnMapping.at(cs::utils::enumCast(type))
                                                .at(cs::utils::enumCast(CatalogColumn::eHipp))],
              hippID);
          if (read && hippID >= 0) {
            continue;
          }
        }

        // store star data
        bool successStoreData(true);

        Star star{};
        successStoreData &= fromString<float>(items[cColumnMapping.at(cs::utils::enumCast(
                                                  type))[cs::utils::enumCast(CatalogColumn::eMag)]],
            star.mMagnitude);
        successStoreData &= fromString<float>(
            items[cColumnMapping.at(
                cs::utils::enumCast(type))[cs::utils::enumCast(CatalogColumn::eRect)]],
            star.mAscension);
        successStoreData &= fromString<float>(
            items[cColumnMapping.at(
                cs::utils::enumCast(type))[cs::utils::enumCast(CatalogColumn::eDecl)]],
            star.mDeclination);

        if (cColumnMapping.at(
                cs::utils::enumCast(type))[cs::utils::enumCast(CatalogColumn::ePara)] > 0) {
          if (!fromString<float>(items[cColumnMapping.at(cs::utils::enumCast(type))
                                           .at(cs::utils::enumCast(CatalogColumn::ePara))],
                  star.mParallax)) {
            star.mParallax = 0;
          }
        } else {
          star.mParallax = 0;
        }

        if (type == CatalogType::eGaia) {
          int   GbpMinusGrpColumn = 6;
          float GbpMinusGrp       = 0;
          successStoreData &= fromString<float>(items[GbpMinusGrpColumn], GbpMinusGrp);

          // https://doi.org/10.1051/0004-6361/201015441
          float logTeff = 3.999F - 0.654F * GbpMinusGrp + 0.709F * std::pow(GbpMinusGrp, 2.F) -
                          0.316F * std::pow(GbpMinusGrp, 3.F);
          star.mTEff = std::pow(10, logTeff);

        } else {
          int   bMagColumn = type == CatalogType::eTycho2 ? 17 : 32;
          float bMag       = 0;
          successStoreData &= fromString<float>(items[bMagColumn], bMag);

          // use B and V magnitude to retrieve the according color
          float bv = bMag - star.mMagnitude;

          // https://arxiv.org/pdf/1201.1809
          // https://github.com/sczesla/PyAstronomy/blob/master/src/pyasl/asl/aslExt_1/ballesterosBV_T.py
          const float t0 = 4600.F;
          const float a  = 0.92F;
          const float b  = 1.7F;
          const float c  = 0.62F;
          star.mTEff     = t0 * (1.0F / (a * bv + b) + 1.0F / (a * bv + c));
        }

        if (successStoreData) {
          star.mAscension   = (360.F + 90.F - star.mAscension) / 180.F * Vista::Pi;
          star.mDeclination = star.mDeclination / 180.F * Vista::Pi;

          mStars.emplace_back(star);
        }
      }

      // Print progress status every 10000 stars.
      if (mStars.size() % 10000 == 0) {
        logger().info("Read {} stars so far...", mStars.size());
      }
    }
    file.close();
    success = true;

    logger().info("Read a total of {} stars.", mStars.size());
  } else {
    logger().error("Failed to load stars: Cannot open catalog file '{}'!", filename);
  }

  return success;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stars::writeStarCache(const std::string& sCacheFile) const {
  VistaType::uint32 catalogs = 0;
  for (auto const& mCatalog : mCatalogs) {
    catalogs += static_cast<uint32_t>(std::pow(2, static_cast<int>(mCatalog.first)));
  }

  VistaByteBufferSerializer serializer;
  serializer.WriteInt32(
      static_cast<VistaType::uint32>(cCacheVersion)); // cache format version number
  serializer.WriteInt32(catalogs);                    // cache format version number
  serializer.WriteInt32(static_cast<VistaType::uint32>(
      mStars.size())); // write number of stars to front of byte stream

  for (const auto& mStar : mStars) {
    // serialize star data into byte stream
    serializer.WriteFloat32(mStar.mMagnitude);
    serializer.WriteFloat32(mStar.mTEff);
    serializer.WriteFloat32(mStar.mAscension);
    serializer.WriteFloat32(mStar.mDeclination);
    serializer.WriteFloat32(mStar.mParallax);
  }

  // open file
  std::ofstream file;
  file.open(sCacheFile.c_str(), std::ios::out | std::ios::binary);
  if (file.is_open()) {
    // write serialized star data
    logger().info("Writing {} stars ({} bytes) into '{}'.", mStars.size(),
        serializer.GetBufferSize(), sCacheFile);

    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    file.write(reinterpret_cast<const char*>(serializer.GetBuffer()), serializer.GetBufferSize());
    file.close();
  } else {
    logger().error(
        "Failed to write binary star data: Cannot open file '{}' for writing!", sCacheFile);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Stars::readStarCache(const std::string& sCacheFile) {
  bool success = false;

  // open file
  std::ifstream file;
  file.open(sCacheFile.c_str(),
      std::ios::in | std::ios::binary | std::ios::ate); // ate = set read pointer to end
  if (file.is_open()) {

    // read binary data from file
    int                          size = static_cast<int>(file.tellg());
    std::vector<VistaType::byte> data(size);

    // set read pointer to the beginning of file stream
    file.seekg(0, std::ios::beg);

    // read file stream into char array 'data'
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    file.read(reinterpret_cast<char*>(&data[0]), size);
    file.close();

    // de-serialize byte stream
    VistaType::uint32 cacheVersion = 0;
    VistaType::uint32 catalogs     = 0;
    VistaType::uint32 numStars     = 0;

    VistaByteBufferDeSerializer deserializer;
    deserializer.SetBuffer(&data[0], size); // prepare for de-serialization
    deserializer.ReadInt32(cacheVersion);   // read cache format version number
    deserializer.ReadInt32(catalogs);       // read which catalogs were loaded
    deserializer.ReadInt32(numStars);       // read number of stars from front of byte stream

    if (cacheVersion != cCacheVersion) {
      return false;
    }

    VistaType::uint32 catalogsToLoad = 0;
    for (const auto& mCatalog : mCatalogs) {
      catalogsToLoad += static_cast<uint32_t>(std::pow(2, static_cast<int>(mCatalog.first)));
    }

    if (catalogs != catalogsToLoad) {
      return false;
    }

    for (unsigned int num = 0; num < numStars; ++num) {
      Star star{};
      deserializer.ReadFloat32(star.mMagnitude);
      deserializer.ReadFloat32(star.mTEff);
      deserializer.ReadFloat32(star.mAscension);
      deserializer.ReadFloat32(star.mDeclination);
      deserializer.ReadFloat32(star.mParallax);

      mStars.emplace_back(star);

      // print progress status
      if (mStars.size() % 100000 == 0) {
        logger().info("Read {} stars so far...", mStars.size());
      }
    }

    success = true;

    logger().info("Read a total of {} stars.", mStars.size());
  }

  return success;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stars::buildStarVAO() {
  int                index(0);
  const int          iElementCount(5);
  std::vector<float> data(iElementCount * mStars.size());

  for (auto it = mStars.begin(); it != mStars.end(); ++it, index += iElementCount) {
    // Distance in parsec --- some have parallax of zero; assume a large distance in those cases.
    float fDist = 1000.F;

    if (it->mParallax > 0.F) {
      fDist = 1000.F / it->mParallax;
    }

    glm::vec3 starPos = glm::vec3(glm::cos(it->mDeclination) * glm::cos(it->mAscension) * fDist,
        glm::sin(it->mDeclination) * fDist,
        glm::cos(it->mDeclination) * glm::sin(it->mAscension) * fDist);

    data[index]     = starPos[0];
    data[index + 1] = starPos[1];
    data[index + 2] = starPos[2];
    data[index + 3] = it->mTEff;
    data[index + 4] = it->mMagnitude - 5.F * std::log10(fDist / 10.F);
  }

  mStarVBO.Bind(GL_ARRAY_BUFFER);
  mStarVBO.BufferData(iElementCount * mStars.size() * sizeof(float), data.data(), GL_STATIC_DRAW);
  mStarVBO.Release();

  // star positions
  mStarVAO.EnableAttributeArray(0);
  mStarVAO.SpecifyAttributeArrayFloat(
      0, 3, GL_FLOAT, GL_FALSE, iElementCount * sizeof(float), 0, &mStarVBO);

  // temperature
  mStarVAO.EnableAttributeArray(1);
  mStarVAO.SpecifyAttributeArrayFloat(
      1, 1, GL_FLOAT, GL_FALSE, iElementCount * sizeof(float), 3 * sizeof(float), &mStarVBO);

  // magnitude
  mStarVAO.EnableAttributeArray(2);
  mStarVAO.SpecifyAttributeArrayFloat(
      2, 1, GL_FLOAT, GL_FALSE, iElementCount * sizeof(float), 4 * sizeof(float), &mStarVBO);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Stars::buildBackgroundVAO() {
  std::vector<float> data(8);
  data[0] = -1;
  data[1] = 1;
  data[2] = 1;
  data[3] = 1;
  data[4] = -1;
  data[5] = -1;
  data[6] = 1;
  data[7] = -1;

  mBackgroundVBO.Bind(GL_ARRAY_BUFFER);
  mBackgroundVBO.BufferData(data.size() * sizeof(float), &(data[0]), GL_STATIC_DRAW);
  mBackgroundVBO.Release();

  // positions
  mBackgroundVAO.EnableAttributeArray(0);
  mBackgroundVAO.SpecifyAttributeArrayFloat(
      0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), 0, &mBackgroundVBO);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::stars
