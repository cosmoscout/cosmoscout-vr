////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2022 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "SimpleObject.hpp"

#include <VistaKernel/GraphicsManager/VistaNodeBridge.h>
#include <VistaKernel/VistaSystem.h>
#include <VistaKernelOpenSGExt/VistaOpenSGMaterialTools.h>

#include "../../../src/cs-core/Settings.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-graphics/GltfLoader.hpp"
//#include "../../../src/cs-graphics/internal/gltfmodel.hpp"
#include "../../../src/cs-utils/convert.hpp"
#include "../../../src/cs-utils/utils.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <utility>

#include "logger.hpp"
#include "utils.hpp"

namespace csp::simpleobjects {

////////////////////////////////////////////////////////////////////////////////////////////////////

SimpleObject::SimpleObject(std::string const& name, Plugin::Settings::SimpleObject const& config,
    VistaSceneGraph* sceneGraph, std::shared_ptr<cs::core::Settings> settings,
    std::shared_ptr<cs::core::SolarSystem> solarSystem)
    : mConfig(std::make_shared<Plugin::Settings::SimpleObject>(config))
    , mSceneGraph(sceneGraph)
    , mSettings(std::move(settings))
    , mSolarSystem(std::move(solarSystem))
    , mModel(std::make_shared<cs::graphics::GltfLoader>(mConfig->mModelFile, mConfig->mEnvironmentMap /*, true*/)) {


  mAnchorObject = mSolarSystem->getBody(mConfig->mAnchorName);

  mAnchor = std::unique_ptr<cs::scene::CelestialAnchorNode>(new cs::scene::CelestialAnchorNode(mSceneGraph->GetRoot(), mSceneGraph->GetNodeBridge(), name, 
                                            mSettings->getAnchorCenter(mConfig->mAnchorName), mSettings->getAnchorFrame(mConfig->mAnchorName)));
  mSettings->initAnchor(*mAnchor, mConfig->mAnchorName);
  mAnchor->setAnchorScale(mConfig->mScale.get());

// set initial position
  auto lngLat = cs::utils::convert::toRadians(mConfig->mLngLat);
  double height = (mAnchorObject->getHeight(lngLat) + mConfig->mElevation.get()) * mSettings->mGraphics.pHeightScale.get();
  mAnchor->setAnchorPosition(cs::utils::convert::toCartesian(lngLat, mAnchorObject->getRadii(), height));
  
  // ground fixed rotation:
  lastSurfaceNormal = cs::utils::convert::lngLatToNormal(cs::utils::convert::toRadians(mConfig->mLngLat));
  qRot = utils::normalToRotation(lastSurfaceNormal);
  mAnchor->setAnchorRotation(qRot);

  mSolarSystem->registerAnchor(mAnchor);
  mSceneGraph->GetRoot()->AddChild(mAnchor.get());
  
  //mModel->setEnableHDR(true);
  //mModel->setRotation(quat);

  mModel->setLightColor(1.0, 1.0, 1.0);
  mModel->attachTo(mSceneGraph, mAnchor.get());

  VistaOpenSGMaterialTools::SetSortKeyOnSubtree(
      mAnchor.get(), static_cast<int>(cs::utils::DrawOrder::eOpaqueItems));


  //auto modelSize = mModel->getBoundingBox()->GetDiagonalLength();
  //logger().info("Model size: {}", modelSize);  

  editEnabled = false;

}

////////////////////////////////////////////////////////////////////////////////////////////////////

SimpleObject::~SimpleObject() {
  mSolarSystem->unregisterAnchor(mAnchor);
  mSceneGraph->GetRoot()->DisconnectChild(mAnchor.get());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void SimpleObject::setSun(std::shared_ptr<const cs::scene::CelestialObject> const& sun) {
  mSun = sun;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string SimpleObject::getName() const { 
  return mAnchor->GetName();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool SimpleObject::getIntersection(
    glm::dvec3 const& /*rayPos*/, glm::dvec3 const& /*rayDir*/, glm::dvec3& /*pos*/) const {
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double SimpleObject::getHeight(glm::dvec2 /*lngLat*/) const {
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////


void SimpleObject::update(/*double tTime, cs::scene::CelestialObserver const& oObs*/) {
  if(editEnabled) { return; }

  //mAnchor->SetIsEnabled(mAnchorObject->getIsInExistence() && mAnchorObject->pVisible.get());

  if( !(mAnchor->getCenterName() == mSolarSystem->getObserver().getCenterName() && mAnchorObject->getIsInExistence()) ) {
    mAnchor->SetIsEnabled(false); 
    return;
  }

  /*VistaBoundingBox bb;
  mModel->getShared()->GetBoundingBox(bb); 
  logger().info("model size: {}", bb.GetDiagonalLength());*/


  auto alpha_obj = atan(mConfig->mDiagonalLength.get() / glm::length(mSolarSystem->getObserver().getAnchorPosition() - mAnchor->getAnchorPosition()));
  //auto pixels_diagonal = sqrt(mSettings->mGuiPosition->mWidthPixel * mSettings->mGuiPosition->mWidthPixel + mSettings->mGuiPosition->mHeightPixel * mSettings->mGuiPosition->mHeightPixel );

  // may need to be adjusted for VR capability
  auto focalLength = mSettings->mGraphics.pFocalLength.get();
  auto diagonal = mSettings->mGraphics.pSensorDiagonal.get();
  auto fovAngle = 2 * atan( diagonal / ( 2 * focalLength) ) * .0005;

  //logger().info("{}: a_img: {}, a_obj: {}, min: {}, enabled: {}", mAnchor->GetName(), alpha_img, alpha_obj, (alpha_img * .0005), (bool)(alpha_obj > alpha_img * .0005));

  // TODO: enable if object would be visible -> better calculation needed
  // Update position and rotation only if distance between observer and object is 100km or less
  if( alpha_obj > fovAngle * .0005 ) {
        
    mAnchor->SetIsEnabled(true);

    // lngLat converted to radians 
    auto lngLat = cs::utils::convert::toRadians(mConfig->mLngLat);

    double height = (mAnchorObject->getHeight(lngLat) + mConfig->mElevation.get()) * mSettings->mGraphics.pHeightScale.get();
    mAnchor->setAnchorPosition(cs::utils::convert::toCartesian(lngLat, mAnchorObject->getRadii(), height));

    
    glm::dvec3 normal;
    
    if(mConfig->mAlignToSurface.get()) {
      normal = utils::getSurfaceNormal(lngLat, mAnchorObject);
    } else {
      normal = cs::utils::convert::lngLatToNormal(lngLat);
    }
    
    auto newRot = utils::normalToRotation(normal) * mConfig->mRotation.get();

    if(newRot != qRot) { 
      qRot = newRot;
      mAnchor->setAnchorRotation(qRot);

      logger().debug("Rotation of \"{}\" has been updated:", mAnchor->GetName());
      //auto normal = cs::utils::convert::lngLatToNormal(lngLat);
      //logger().info("  normal  x:{}, y:{}, z:{}", normal.x, normal.y, normal.z);
      logger().debug("   Normal x:{}, y:{}, z:{}", normal.x, normal.y, normal.z);
      logger().debug("   Rotation w:{}, x:{}, y:{}, z:{}", qRot.w, qRot.x, qRot.y, qRot.z);
      
      if(std::acos(glm::dot(lastSurfaceNormal, normal)) > 1.13 /* 65° in radians */ ) {
        logger().warn("The rotation of \"{}\" has changed significantly. Maybe this should be verified.", mAnchor->GetName());
      }  

      lastSurfaceNormal = normal;
    }
  } else {
    mAnchor->SetIsEnabled(false);
    //logger().info("{}: disabled", mAnchor->GetName());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////


void SimpleObject::setEditEnabled(bool enabled) {

  if(enabled) editEnabled = true;
  else editEnabled = false;

  if(mAnchor != nullptr) {
    mAnchor->SetIsEnabled(!enabled);
  }
}

bool SimpleObject::isEditedEnabled() const { 
  return editEnabled;
}


void SimpleObject::updateConfig(std::string const& name, Plugin::Settings::SimpleObject const& config) {

  // update config and reinitialize model

  std::shared_ptr<Plugin::Settings::SimpleObject> newConfig = std::make_shared<Plugin::Settings::SimpleObject>(config); 

  if(name != getName()) {
    mAnchor->SetName(name);
  }

  if (newConfig->mModelFile != mConfig->mModelFile || 
      newConfig->mEnvironmentMap != mConfig->mEnvironmentMap || 
      newConfig->mAnchorName != mConfig->mAnchorName) {
    
    mSolarSystem->unregisterAnchor(mAnchor);
    mSceneGraph->GetRoot()->DisconnectChild(mAnchor.get());

    //mAnchor.reset();
    //mModel.reset();


    mAnchorObject = mSolarSystem->getBody(newConfig->mAnchorName);
    mModel = std::make_shared<cs::graphics::GltfLoader>(newConfig->mModelFile, newConfig->mEnvironmentMap /*, true*/);

    mAnchor = std::unique_ptr<cs::scene::CelestialAnchorNode>(new cs::scene::CelestialAnchorNode(mSceneGraph->GetRoot(), mSceneGraph->GetNodeBridge(), getName(), 
                                              mSettings->getAnchorCenter(newConfig->mAnchorName), mSettings->getAnchorFrame(newConfig->mAnchorName)));
                                              
    mSettings->initAnchor(*mAnchor, newConfig->mAnchorName);
    mAnchor->setAnchorScale(newConfig->mScale.get());


    // set initial position
    auto lngLat = cs::utils::convert::toRadians(newConfig->mLngLat);
    double height = (mAnchorObject->getHeight(lngLat) + newConfig->mElevation.get()) * mSettings->mGraphics.pHeightScale.get();
    mAnchor->setAnchorPosition(cs::utils::convert::toCartesian(lngLat, mAnchorObject->getRadii(), height));
    
    // ground fixed rotation:
    lastSurfaceNormal = cs::utils::convert::lngLatToNormal(cs::utils::convert::toRadians(newConfig->mLngLat));
    qRot = utils::normalToRotation(lastSurfaceNormal);
    mAnchor->setAnchorRotation(qRot);


    mSolarSystem->registerAnchor(mAnchor);
    mSceneGraph->GetRoot()->AddChild(mAnchor.get());

    mModel->setLightColor(1.0, 1.0, 1.0);
    mModel->attachTo(mSceneGraph, mAnchor.get());

  }

  if(newConfig->mScale != mConfig->mScale) {
    mAnchor->setAnchorScale(newConfig->mScale.get());
  }

  /*if(name != getName()) {

    mSolarSystem->unregisterAnchor(mAnchor);
    mSceneGraph->GetRoot()->DisconnectChild(mAnchor.get());*/

    // mModel = std::make_shared<cs::graphics::GltfLoader>(mConfig->mModelFile, mConfig->mEnvironmentMap /*, true*/);
  
    // mAnchor = std::unique_ptr<cs::scene::CelestialAnchorNode>(new cs::scene::CelestialAnchorNode(mSceneGraph->GetRoot(), mSceneGraph->GetNodeBridge(), name, 
    //                                           mSettings->getAnchorCenter(mConfig->mAnchorName), mSettings->getAnchorFrame(mConfig->mAnchorName)));
                                              
    // mSettings->initAnchor(*mAnchor, mConfig->mAnchorName);
    

  //}


//   auto lngLat = cs::utils::convert::toRadians(mConfig->mLngLat);

//   if(newConfig->mLngLat != mConfig->mLngLat) { 
//     double height = (mAnchorObject->getHeight(lngLat) + mConfig->mElevation.get()) * mSettings->mGraphics.pHeightScale.get();
//     mAnchor->setAnchorPosition(cs::utils::convert::toCartesian(lngLat, mAnchorObject->getRadii(), height));  
//   }

// // set initial position
  
  
//   // ground fixed rotation:
//   if(newConfig->mRotation != mConfig->mRotation) {
//     lastSurfaceNormal = cs::utils::convert::lngLatToNormal(cs::utils::convert::toRadians(mConfig->mLngLat));
//     qRot = utils::normalToRotation(lastSurfaceNormal);
//     mAnchor->setAnchorRotation(qRot);

//   }

  mConfig = std::move(newConfig);

}



} // namespace csp::simpleobjects
