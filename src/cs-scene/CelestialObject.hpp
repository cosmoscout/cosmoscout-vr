////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_SCENE_CELESTIAL_OBJECT_HPP
#define CS_SCENE_CELESTIAL_OBJECT_HPP

#include "CelestialAnchor.hpp"

#include <limits>
#include <memory>

namespace cs::scene {

class CelestialBody;

/// CelestialObjects have a lifetime in the universe. They are defined by their start and end
/// existence time. The time is given in the Barycentric Dynamical Time format, which is used
/// throughout SPICE.
class CS_SCENE_EXPORT CelestialObject : public CelestialAnchor {
 public:
  explicit CelestialObject(
      std::string sCenterName = "Solar System Barycenter", std::string sFrameName = "J2000");

  CelestialObject(CelestialObject const& other) = default;
  CelestialObject(CelestialObject&& other)      = default;

  CelestialObject& operator=(CelestialObject const& other) = default;
  CelestialObject& operator=(CelestialObject&& other) = default;

  virtual ~CelestialObject() = default;

  // -----------------------------------------------------------------------------------------------

  /// The time range in Barycentric Dynamical Time in which the object existed.
  /// This should match the time coverage of the loaded SPICE kernels.
  glm::dvec2 const& getExistence() const;
  void              setExistence(glm::dvec2 value);

  /// The radii of the CelestialBody in meters.
  glm::dvec3 const& getRadii() const;
  void              setRadii(glm::dvec3 const& value);

  /// An approximate radius of the body in meters. This will serve as a basis for visibility
  /// calculation. If set to 0.0 (the default), getIsBodyVisible() will always return true.
  double getBodyCullingRadius() const;
  void   setBodyCullingRadius(double value);

  /// An approximate radius of the bodies orbit in meters. This will serve as a basis for visibility
  /// calculation of the bodies trajectory. If set to 0.0 (the default), getIsOrbitVisible() will
  /// always return true.
  double getOrbitCullingRadius() const;
  void   setOrbitCullingRadius(double value);

  /// If this is set to true, the observer may follow this body on its path through the Solar
  /// System.
  bool getIsTrackable() const;
  void setIsTrackable(bool value);

  /// If this is set to true, the observer will collide with the surface of this body. The surface
  /// is modelled as an ellipsoid displaced according to getHeight() along the geodetic surface
  /// normal.
  bool getIsCollidable() const;
  void setIsCollidable(bool value);

  // -----------------------------------------------------------------------------------------------

  /// This is called once a frame by the SolarSystem if this CelestialObject has been registered
  /// with the SolarSystem. This will update all time- and observer-dependent members. These are the
  /// observer-centric transformation, the result of getIsInExistence(), getIsBodyVisible(), and
  /// getIsOrbitVisible().
  virtual void update(double tTime, CelestialObserver const& oObs);

  /// @return true, if the current time is in between the start and end existence values.
  bool getIsInExistence() const;

  /// @return true, if the current distance to the observer suggests that the body could be visible
  /// (this is based on mBodyCullingRadius and updated during update()).
  bool getIsBodyVisible() const;

  /// @return true, if the current distance to the observer suggests that the bodies trajectory
  /// could be visible (this is based on mOrbitCullingRadius and updated during update()).
  bool getIsOrbitVisible() const;

  /// Returns the current relative transformation to the observer.
  glm::dmat4 const& getObserverRelativeTransform() const;
  glm::dvec4        getObserverRelativePosition() const;

  // -----------------------------------------------------------------------------------------------

  /// It is possible to assign a CelestialBody to a CelestialObject. This body can be used to define
  /// an actual surface by providing a getHeight() method.
  std::shared_ptr<CelestialBody> const& getBody() const;
  void                                  setBody(std::shared_ptr<CelestialBody> const& body);

 protected:
  glm::dvec3 mRadii = glm::dvec3(0.0);

  double mBodyCullingRadius  = 0.0;
  double mOrbitCullingRadius = 0.0;
  bool   mIsTrackable        = true;
  bool   mIsCollidable       = true;

  glm::dvec2 mExistence =
      glm::dvec2(std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max());

  glm::dmat4 matObserverRelativeTransform = glm::dmat4(1.0);
  bool       mIsInExistence               = false;
  bool       mIsBodyVisible               = true;
  bool       mIsOrbitVisible              = true;

  std::shared_ptr<CelestialBody> mBody;
};

} // namespace cs::scene

#endif // CS_SCENE_CELESTIAL_OBJECT_HPP
