////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_SCENE_CELESTIAL_OBJECT_HPP
#define CS_SCENE_CELESTIAL_OBJECT_HPP

#include "CelestialAnchor.hpp"

#include <array>
#include <memory>
#include <optional>

namespace cs::scene {

class CelestialSurface;
class CelestialObserver;
class IntersectableObject;

/// CelestialObjects are configured in the scene configuration file and instantiated by the Settings
/// class. They are updated once each frame by the SolarSystem. Plugins can get references to
/// CelestialObjects via SolarSystem::getObject() and retrieve their current observer-centric
/// transformation each frame.
/// CelestialObjects have a lifetime in the universe. They are defined by
/// their start and end existence time. The time is given in the Barycentric Dynamical Time format,
/// which is used throughout SPICE.
class CS_SCENE_EXPORT CelestialObject : public CelestialAnchor {
 public:
  explicit CelestialObject(
      std::string sCenterName = "Solar System Barycenter", std::string sFrameName = "J2000");

  CelestialObject(CelestialObject const& other) = default;
  CelestialObject(CelestialObject&& other)      = default;

  CelestialObject& operator=(CelestialObject const& other) = default;
  CelestialObject& operator=(CelestialObject&& other) = default;

  ~CelestialObject() override = default;

  // -----------------------------------------------------------------------------------------------

  /// The time range in Barycentric Dynamical Time in which the object existed.
  /// This should match the time coverage of the loaded SPICE kernels.
  glm::dvec2 getExistence() const;
  void       setExistence(glm::dvec2 value);

  /// Returns the existence of the object encoded in strings in the format YYYY-MM-DDTHH:MM:SS.fffZ.
  /// This is used for serializing the objects into the json configuration files.
  std::array<std::string, 2> getExistenceAsStrings() const;
  void                       setExistenceAsStrings(std::array<std::string, 2> value);

  /// The radii of the CelestialObject in meters. If setRadii() was never called, this method
  /// will attempt to get the radii from SPICE once. If this fails, the method will
  /// return [0.0, 0.0, 0.0].
  glm::dvec3 const& getRadii() const;
  void              setRadii(glm::dvec3 value);

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

  /// This is overridden to reset the body radii which are obtained from SPICE if no radii have been
  /// set with setRadii().
  void setCenterName(std::string sCenterName) override;

  // -----------------------------------------------------------------------------------------------

  /// This is called once a frame by the SolarSystem if this CelestialObject has been registered
  /// with the SolarSystem. This will update all time- and observer-dependent members. These are the
  /// observer-centric transformation, the result of getIsInExistence(), getIsBodyVisible(), and
  /// getIsOrbitVisible().
  void update(double tTime, CelestialObserver const& oObs) const;

  /// @return true, if the current time is in between the start and end existence values.
  bool getIsInExistence() const;

  /// @return true, if the latest call to update() achieved to get a valid observer-relative
  /// transformation.
  bool getHasValidPosition() const;

  /// @return true, if the current distance to the observer suggests that the body could be visible
  /// (this is based on mBodyCullingRadius and updated during update()). This will also return false
  /// if either getIsInExistence() or getHasValidPosition() returns false.
  bool getIsBodyVisible() const;

  /// @return true, if the current distance to the observer suggests that the bodies trajectory
  /// could be visible (this is based on mOrbitCullingRadius and updated during update()). This will
  /// also return false if either getIsInExistence() or getHasValidPosition() returns false.
  bool getIsOrbitVisible() const;

  /// Returns the current relative transformation to the observer. This is the matrix which
  /// transforms objects from the observer's coordinate system to the CelestialObject's coordinate
  /// system. This usually changes during a call to update().
  glm::dmat4 const& getObserverRelativeTransform() const;

  /// This is the same as above, however it does not return the full transformation but only the
  /// observer-relative position (ignoring the rotation and scale of this).
  glm::dvec3 getObserverRelativePosition() const;

  /// This is a convenience method to compute the current observer-relative transformation of an
  /// object which has a certain transformation relative to this CelestialObject. For example, this
  /// method could be used to get the observer-relative transformation of an object on the surface
  /// of a planet.
  glm::dmat4 getObserverRelativeTransform(glm::dvec3 const& translation,
      glm::dquat const& rotation = glm::dquat(1.0, 0.0, 0.0, 0.0), double scale = 1.0) const;

  /// This is the same as above, however it does not return the full transformation of the
  /// sub-object but only the observer-relative position (ignoring the rotation and scale of the
  /// sub-object).
  glm::dvec3 getObserverRelativePosition(glm::dvec3 const& translation) const;

  // -----------------------------------------------------------------------------------------------

  /// It is possible to assign a CelestialSurface to a CelestialObject. This surface can be used to
  /// define an actual terrain by providing a getHeight() method. This will be used for ground
  /// following, collision detection and by plugins to sample the height of the body (for instance
  /// for measuring tools). Assigning a CelestialSurface does not change the state of the
  /// CelestialObject itself, therefore this method is considered to be const.
  std::shared_ptr<CelestialSurface> const& getSurface() const;
  void setSurface(std::shared_ptr<CelestialSurface> surface) const;

  /// It is also possible to assign an IntersectableObject to a CelestialObject. If the
  /// CelestialObject is then registered with the InputManager, it will be regularly tested for
  /// intersections with the mouse ray. Assigning an IntersectableObject does not change the state
  /// of the CelestialObject itself, therefore this method is considered to be const.
  std::shared_ptr<IntersectableObject> const& getIntersectableObject() const;
  void setIntersectableObject(std::shared_ptr<IntersectableObject> object) const;

 protected:
  glm::dvec3 mRadii              = glm::dvec3(0.0);
  double     mBodyCullingRadius  = 0.0;
  double     mOrbitCullingRadius = 0.0;
  bool       mIsTrackable        = true;
  bool       mIsCollidable       = true;

  // These are mutable since they are assigned lazily in the const getExistence() or
  // getExistenceAsStrings() methods.
  mutable std::optional<glm::dvec2>                 mExistence;
  mutable std::optional<std::array<std::string, 2>> mExistenceAsStrings;

  // These are mutable since assigning a CelestialSurface or an IntersectableObject does not change
  // the state of the CelestialObject itself.
  mutable std::shared_ptr<CelestialSurface>    mSurface;
  mutable std::shared_ptr<IntersectableObject> mIntersectable;

  // This is mutable because the celestial will try to get its radii from SPICE in the first call to
  // the otherwise const method getRadii().
  mutable glm::dvec3 mRadiiFromSPICE = glm::dvec3(-1.0);

  // These members are mutable as they have to be changed in the const update() method. They do not
  // represent properties of the object but rather the cached observer-relative state.
  mutable glm::dmat4 matObserverRelativeTransform = glm::dmat4(1.0);
  mutable bool       mIsInExistence               = false;
  mutable bool       mIsBodyVisible               = false;
  mutable bool       mIsOrbitVisible              = false;
  mutable bool       mHasValidPosition            = false;
};

} // namespace cs::scene

#endif // CS_SCENE_CELESTIAL_OBJECT_HPP
