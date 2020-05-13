////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "VistaPlanet.hpp"

#include "TileSource.hpp"
#include "UpdateBoundsVisitor.hpp"

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
VistaPlanet::VistaPlanet(std::shared_ptr<GLResources> const& glResources)
    : mWorldTransform(1.0)
    , mLodVisitor(mParams)
    , mRenderer(mParams)
    , mSrcDEM(nullptr)
    , mTreeMgrDEM(mParams, glResources)
    , mSrcIMG(nullptr)
    , mTreeMgrIMG(mParams, glResources)
    , mLastFrameClock(GetVistaSystem()->GetFrameClock())
    , mSumFrameClock(0.0)
    , mSumDrawTiles(0)
    , mSumLoadTiles(0)
    , mMaxDrawTiles(0)
    , mMaxLoadTiles(0)
    , mFlags(0) {
  mTreeMgrDEM.setName("DEM");
  mTreeMgrIMG.setName("IMG");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* virtual */
VistaPlanet::~VistaPlanet() {
  // clear tree managers
  mTreeMgrDEM.clear();
  mTreeMgrIMG.clear();

  if (mSrcDEM) {
    mSrcDEM->fini();
  }

  if (mSrcIMG) {
    mSrcIMG->fini();
  }

#if !defined(NDEBUG) && !defined(VISTAPLANET_NO_VERBOSE)
  vstr::outi() << "[VistaPlanet::~VistaPlanet] maxDrawTiles " << mMaxDrawTiles << " maxLoadTiles "
               << mMaxLoadTiles << std::endl;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* virtual */ void VistaPlanet::doShadows() {
  mLodVisitor.visit();

  // get matrices and viewport
  int          frameCount = GetVistaSystem()->GetFrameLoop()->GetFrameCount();
  glm::dmat4   matVM      = getModelviewMatrix();
  glm::fmat4x4 matP       = getProjectionMatrix();
  renderTiles(frameCount, matVM, matP, nullptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool VistaPlanet::getWorldTransform(VistaTransformMatrix& matTransform) const {
  matTransform = VistaTransformMatrix();
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* virtual */ bool VistaPlanet::Do() {
  doFrame();

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* virtual */ bool VistaPlanet::GetBoundingBox(VistaBoundingBox& bb) {
  // XXX TODO use actual values!!
  //     Turns out ViSTA never calls this function, so it does not really
  //     matter..
  //     -- neume 2014-03-12
  VistaVector3D bbMin(0.F, 0.F, 0.F);
  VistaVector3D bbMax(0.F, 0.F, 0.F);

  bb.SetBounds(bbMin, bbMax);

  return true;
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

void VistaPlanet::setTerrainShader(TerrainShader* shader) {
  mRenderer.setTerrainShader(shader);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TerrainShader* VistaPlanet::getTerrainShader() const {
  return mRenderer.getTerrainShader();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::setDEMSource(TileSource* srcDEM) {
  // Don't do anything if nothing changed.
  if (mSrcDEM == srcDEM) {
    return;
  }

  // shut down old source
  if (mSrcDEM) {
    mSrcDEM->fini();

    mLodVisitor.setTreeManagerDEM(nullptr);
    mRenderer.setTreeManagerDEM(nullptr);
    mTreeMgrDEM.setSource(nullptr);
  }

  mSrcDEM = srcDEM;

  // init new source
  if (mSrcDEM) {
    mSrcDEM->init();

    mTreeMgrDEM.setSource(mSrcDEM);
    mLodVisitor.setTreeManagerDEM(&mTreeMgrDEM);
    mRenderer.setTreeManagerDEM(&mTreeMgrDEM);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TileSource* VistaPlanet::getDEMSource() const {
  return mSrcDEM;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::setIMGSource(TileSource* srcIMG) {
  // Don't do anything if nothing changed.
  if (mSrcIMG == srcIMG) {
    return;
  }

  // shut down old source
  if (mSrcIMG) {
    mSrcIMG->fini();

    mLodVisitor.setTreeManagerIMG(nullptr);
    mRenderer.setTreeManagerIMG(nullptr);
    mTreeMgrIMG.setSource(nullptr);
  }

  mSrcIMG = srcIMG;

  // init new source
  if (mSrcIMG) {
    mSrcIMG->init();

    mTreeMgrIMG.setSource(mSrcIMG);
    mLodVisitor.setTreeManagerIMG(&mTreeMgrIMG);
    mRenderer.setTreeManagerIMG(&mTreeMgrIMG);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TileSource* VistaPlanet::getIMGSource() const {
  return mSrcIMG;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// The funciton that drives all operations that need to be done each frame.
// It simply calls the other functions in this section in order and passes
// a few shared values between them (e.g. the matrices for the current view).
void VistaPlanet::doFrame() {
  int frameCount = GetVistaSystem()->GetFrameLoop()->GetFrameCount();

  // get matrices and viewport
  glm::dmat4   matVM    = getModelviewMatrix();
  glm::fmat4x4 matP     = getProjectionMatrix();
  glm::ivec4   viewport = getViewport();

  // collect/print statistics
  updateStatistics(frameCount);

  // update bounding boxes
  updateTileBounds();

  // integrate newly loaded tiles/remove unused tiles
  updateTileTrees(frameCount);

  // determine tiles to draw and load
  traverseTileTrees(frameCount, matVM, matP, viewport);

  // pass requests to load tiles to TreeManagers
  processLoadRequests();

  // render
  renderTiles(frameCount, matVM, matP, mShadowMap);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::updateStatistics(int frameCount) {
  double frameClock = GetVistaSystem()->GetFrameClock();
  double frameT     = (frameClock - mLastFrameClock);
  mLastFrameClock   = frameClock;

  // update statistics
  mMaxDrawTiles = std::max(mMaxDrawTiles,
      std::max(mLodVisitor.getRenderDEM().size(), mLodVisitor.getRenderIMG().size()));
  mMaxLoadTiles =
      std::max(mMaxLoadTiles, mLodVisitor.getLoadDEM().size() + mLodVisitor.getLoadIMG().size());

  // running sums of frame time, tiles to draw, and tiles to load
  // used below to calculate averages every 60 frames
  mSumFrameClock += frameT;
  mSumDrawTiles += std::max(mLodVisitor.getRenderDEM().size(), mLodVisitor.getRenderIMG().size());
  mSumLoadTiles += mLodVisitor.getLoadDEM().size() + mLodVisitor.getLoadIMG().size();

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

void VistaPlanet::updateTileBounds() {
  if ((mFlags & sFlagTileBoundsInvalid) != 0) {
    // rebuild bounding boxes
    UpdateBoundsVisitor ubVisitor(&mTreeMgrDEM, mParams);
    ubVisitor.visit();

    mFlags &= ~sFlagTileBoundsInvalid;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::updateTileTrees(int frameCount) {
  // update DEM tree
  if (mSrcDEM) {
    mTreeMgrDEM.setFrameCount(frameCount);
    mTreeMgrDEM.update();
  }

  // update IMG tree
  if (mSrcIMG) {
    mTreeMgrIMG.setFrameCount(frameCount);
    mTreeMgrIMG.update();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::traverseTileTrees(
    int frameCount, glm::dmat4 const& matVM, glm::fmat4x4 const& matP, glm::ivec4 const& viewport) {
  // update per-frame information of LODVisitor
  mLodVisitor.setFrameCount(frameCount);
  mLodVisitor.setModelview(matVM);
  mLodVisitor.setProjection(matP);
  mLodVisitor.setViewport(viewport);

  // traverse quad trees and determine nodes to render and load
  // respectively
  mLodVisitor.visit();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::processLoadRequests() {
  if (mSrcDEM) {
    mTreeMgrDEM.request(mLodVisitor.getLoadDEM());
  }

  if (mSrcIMG) {
    mTreeMgrIMG.request(mLodVisitor.getLoadIMG());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::renderTiles(int frameCount, glm::dmat4 const& matVM, glm::fmat4x4 const& matP,
    cs::graphics::ShadowMap* shadowMap) {
  // update per-frame information of TileRenderer
  mRenderer.setFrameCount(frameCount);
  mRenderer.setModelview(matVM);
  mRenderer.setProjection(matP);
  mRenderer.render(mLodVisitor.getRenderDEM(), mLodVisitor.getRenderIMG(), shadowMap);
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
glm::dmat4 VistaPlanet::getModelviewMatrix() const {
  std::array<GLfloat, 16> glMat{};
  glGetFloatv(GL_MODELVIEW_MATRIX, glMat.data());

  return glm::dmat4(glm::make_mat4x4(glMat.data())) * mWorldTransform;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dmat4 VistaPlanet::getProjectionMatrix() {
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

void VistaPlanet::setEquatorialRadius(float radius) {
  mParams.mEquatorialRadius = radius;

  mFlags |= sFlagTileBoundsInvalid;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double VistaPlanet::getEquatorialRadius() const {
  return mParams.mEquatorialRadius;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::setPolarRadius(float radius) {
  mParams.mPolarRadius = radius;

  mFlags |= sFlagTileBoundsInvalid;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double VistaPlanet::getPolarRadius() const {
  return mParams.mPolarRadius;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VistaPlanet::setHeightScale(float scale) {
  mParams.mHeightScale = scale;

  mFlags |= sFlagTileBoundsInvalid;
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
