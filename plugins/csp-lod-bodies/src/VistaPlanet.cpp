////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "VistaPlanet.hpp"

#include "../../../src/cs-utils/FrameStats.hpp"
#include "../../../src/cs-utils/utils.hpp"
#include "TileSource.hpp"

#include <VistaBase/VistaStreamUtils.h>
#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaDisplaySystem.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaKernel/VistaFrameLoop.h>
#include <VistaKernel/VistaSystem.h>
#include <glm/gtc/type_ptr.hpp>

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ bool VistaPlanet::sGlewInitialized = false;

////////////////////////////////////////////////////////////////////////////////////////////////////

/* explicit */
VistaPlanet::VistaPlanet(std::shared_ptr<GLResources> glResources, uint32_t tileResolution)
    : mWorldTransform(1.0)
    , mTreeMgr(std::move(glResources))
    , mLodVisitor(mParams, &mTreeMgr)
    , mRenderer(mParams, &mTreeMgr, tileResolution)
    , mLastFrameClock(GetVistaSystem()->GetFrameClock())
    , mSumFrameClock(0.0)
    , mSumDrawTiles(0)
    , mSumLoadTiles(0)
    , mMaxDrawTiles(0)
    , mMaxLoadTiles(0)
    , mFlags(0) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* virtual */
VistaPlanet::~VistaPlanet() {
  // clear tree managers
  mTreeMgr.clear();

  for (auto* src : mTileDataSources.mChannels) {
    src->fini();
  }

#if !defined(NDEBUG) && !defined(VISTAPLANET_NO_VERBOSE)
  vstr::outi() << "[VistaPlanet::~VistaPlanet] maxDrawTiles " << mMaxDrawTiles << " maxLoadTiles "
               << mMaxLoadTiles << std::endl;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// The function that drives all operations that need to be done each frame.
// It simply calls the other functions in this section in order and passes
// a few shared values between them (e.g. the matrices for the current view).
void VistaPlanet::draw() {
  if (!mEnabled) {
    return;
  }

  int frameCount = GetVistaSystem()->GetFrameLoop()->GetFrameCount();

  // get matrices and viewport
  glm::mat4  matV     = getViewMatrix();
  glm::mat4  matP     = getProjectionMatrix();
  glm::ivec4 viewport = getViewport();

  // collect/print statistics
  updateStatistics(frameCount);

  // integrate newly loaded tiles/remove unused tiles
  {
    cs::utils::FrameStats::ScopedTimer timer(
        "Update Tile Trees", cs::utils::FrameStats::TimerMode::eCPU);
    updateTileTrees(frameCount);
  }

  // determine tiles to draw and load
  {
    cs::utils::FrameStats::ScopedTimer timer(
        "Traverse Tile Trees", cs::utils::FrameStats::TimerMode::eCPU);
    traverseTileTrees(frameCount, mWorldTransform, matV, matP, viewport);
  }

  // pass requests to load tiles to TreeManagers
  {
    cs::utils::FrameStats::ScopedTimer timer(
        "Rrocess Load Requests", cs::utils::FrameStats::TimerMode::eCPU);
    processLoadRequests();
  }

  // render
  {
    cs::utils::FrameStats::ScopedTimer timer(
        "Render Tiles", cs::utils::FrameStats::TimerMode::eCPU);
    renderTiles(frameCount, mWorldTransform, matV, matP, mShadowMap);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::drawForShadowMap() {
  if (!mEnabled) {
    return;
  }

  // get matrices and viewport
  int          frameCount = GetVistaSystem()->GetFrameLoop()->GetFrameCount();
  glm::dmat4   matV       = getViewMatrix();
  glm::fmat4x4 matP       = getProjectionMatrix();
  glm::ivec4   viewport   = getViewport();

  traverseTileTrees(frameCount, mWorldTransform, matV, matP, viewport);

  renderTiles(frameCount, mWorldTransform, matV, matP, nullptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::setWorldTransform(glm::dmat4 const& mat) {
  mWorldTransform = mat;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dmat4 VistaPlanet::getWorldTransform() const {
  return mWorldTransform;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::setEnabled(bool enabled) {
  mEnabled = enabled;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool VistaPlanet::getEnabled() const {
  return mEnabled;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::setTerrainShader(TerrainShader* shader) {
  mRenderer.setTerrainShader(shader);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TerrainShader* VistaPlanet::getTerrainShader() const {
  return mRenderer.getTerrainShader();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::setDataSource(TileDataType type, TileSource* src) {
  // Don't do anything if nothing changed.
  if (mTileDataSources.get(type) == src) {
    return;
  }

  // shut down old source
  if (mTileDataSources.get(type)) {
    mTileDataSources.get(type)->fini();
    mTreeMgr.setSource(type, nullptr);
  }
  mTileDataSources.set(type, src);

  // init new source
  if (src) {
    src->init();
    mTreeMgr.setSource(type, src);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TileSource* VistaPlanet::getDataSource(TileDataType type) const {
  return mTileDataSources.get(type);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::updateStatistics(int frameCount) {
  double frameClock = GetVistaSystem()->GetFrameClock();
  double frameT     = (frameClock - mLastFrameClock);
  mLastFrameClock   = frameClock;

  // update statistics
  mMaxDrawTiles = std::max(mMaxDrawTiles, mLodVisitor.getRenderNodes().size());
  mMaxLoadTiles = std::max(mMaxLoadTiles, mLodVisitor.getLoadNodes().size());

  // running sums of frame time, tiles to draw, and tiles to load
  // used below to calculate averages every 60 frames
  mSumFrameClock += frameT;
  mSumDrawTiles += mLodVisitor.getRenderNodes().size();
  mSumLoadTiles += mLodVisitor.getLoadNodes().size();

  // print and reset statistics every 60 frames
  if (frameCount % 60 == 0) {
#if !defined(NDEBUG) && !defined(VISTAPLANET_NO_VERBOSE)
    vstr::outi() << "[VistaPlanet::Do] frame [" << vstr::framecount << "] avg. fps ["
                 << std::setprecision(2) << std::setw(4) << (60.0 / mSumFrameClock)
                 << "] avg. frameclock [" << std::setprecision(3) << std::setw(4)
                 << (mSumFrameClock / 60.0) << "] avg. draw tiles [" << (mSumDrawTiles / 60.0)
                 << "] avg. load tiles [" << (mSumLoadTiles / 60.0) << "]"
                 << std::setprecision(6); // reset to default

    if ((mSumFrameClock / 60.0) > 0.017) {
      vstr::out() << " -- frame budget exceeded!";
    }

    vstr::out() << std::endl;
#endif

    mSumFrameClock = 0.0;
    mSumDrawTiles  = 0;
    mSumLoadTiles  = 0;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::updateTileTrees(int frameCount) {
  cs::utils::FrameStats::ScopedTimer timer("Upload Tiles", cs::utils::FrameStats::TimerMode::eCPU);
  mTreeMgr.setFrameCount(frameCount);
  mTreeMgr.update();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::traverseTileTrees(int frameCount, glm::dmat4 const& matM, glm::mat4 const& matV,
    glm::mat4 const& matP, glm::ivec4 const& viewport) {
  // update per-frame information of LODVisitor
  mLodVisitor.setFrameCount(frameCount);
  mLodVisitor.setModelview(glm::dmat4(matV) * matM);
  mLodVisitor.setProjection(matP);
  mLodVisitor.setViewport(viewport);

  // traverse quad trees and determine nodes to render and load
  // respectively
  mLodVisitor.visit();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::processLoadRequests() {
  mTreeMgr.request(mLodVisitor.getLoadNodes());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::renderTiles(int frameCount, glm::dmat4 const& matM, glm::mat4 const& matV,
    glm::mat4 const& matP, cs::graphics::ShadowMap* shadowMap) {
  // update per-frame information of TileRenderer
  mRenderer.setFrameCount(frameCount);
  mRenderer.setModel(matM);
  mRenderer.setView(matV);
  mRenderer.setProjection(matP);
  mRenderer.render(mLodVisitor.getRenderNodes(), shadowMap);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// helper functions

// It appears that ViSTA does not give access to the modelview and
// projection matrices easily.
// While it is possible to get the "view" matrix via
// VistaDisplaySystem->GetReferenceFrame() that does not contain
// the initial viewer position set through the .ini file VIEWER_POSITION
// setting.
// Also, an IVistaOpenGLDraw object can not easily obtain the scene graph
// node it is attached to, in order to find its model matrix.
//
// Similarly the projection obtained from
// VistaDisplaySystem->GetViewport(0)->GetProjection()
// does not match the contents of GL_PROJECTION_MATRIX.
//
// If there is a better (i.e. using ViSTA interfaces) way to obtain these
// matrices replace the implementation of getModelviewMatrix() and
// getProjectionMatrix().
glm::mat4 VistaPlanet::getViewMatrix() const {
  std::array<GLfloat, 16> glMat{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMat.data());

  return glm::make_mat4x4(glMat.data());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::mat4 VistaPlanet::getProjectionMatrix() {
  std::array<GLfloat, 16> glMat{};
  glGetFloatv(GL_PROJECTION_MATRIX, glMat.data());

  return glm::make_mat4x4(glMat.data());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::ivec4 VistaPlanet::getViewport() {
  std::array<GLint, 4> glVP{};
  glGetIntegerv(GL_VIEWPORT, glVP.data());

  return glm::make_vec4(glVP.data());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::setRadii(glm::dvec3 const& radii) {
  if (mParams.mRadii != radii) {
    mParams.mRadii = radii;
    mLodVisitor.queueRecomputeTileBounds();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 const& VistaPlanet::getRadii() const {
  return mParams.mRadii;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::setHeightScale(float scale) {
  if (mParams.mHeightScale != scale) {
    mParams.mHeightScale = scale;
    mLodVisitor.queueRecomputeTileBounds();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double VistaPlanet::getHeightScale() const {
  return mParams.mHeightScale;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::setLODFactor(float lodFactor) {
  mParams.mLodFactor = lodFactor;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double VistaPlanet::getLODFactor() const {
  return mParams.mLodFactor;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::setMinLevel(int minLevel) {
  mParams.mMinLevel = minLevel;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int VistaPlanet::getMinLevel() const {
  return mParams.mMinLevel;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::setMaxLevel(int maxLevel) {
  mParams.mMaxLevel = maxLevel;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int VistaPlanet::getMaxLevel() const {
  return mParams.mMaxLevel;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TileRenderer& VistaPlanet::getTileRenderer() {
  return mRenderer;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TileRenderer const& VistaPlanet::getTileRenderer() const {
  return mRenderer;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

LODVisitor& VistaPlanet::getLODVisitor() {
  return mLodVisitor;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

LODVisitor const& VistaPlanet::getLODVisitor() const {
  return mLodVisitor;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
