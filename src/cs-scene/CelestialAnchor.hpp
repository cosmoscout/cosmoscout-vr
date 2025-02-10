////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_SCENE_CELESTIAL_ANCHOR_HPP
#define CS_SCENE_CELESTIAL_ANCHOR_HPP

#include "cs_scene_export.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <string>

namespace cs::scene {

/// This class is the root class for all objects which have a location in the Solar System. It uses
/// the SPICE coordinate system, which means that the transformation of this object is defined by
/// two parameters:
/// - The center of the coordinate system:
///    This is the origin of the coordinate system. The transformation of the object describes the
///    difference from the objects location relative this center.
///    An example for this could be the center of the Earth.
/// - The frame of reference:
///    This defines how the position is calculated relative to the origin. For example with
///    "J2000" the location would be calculated relative to the equatorial plane, but the rotation
///    of the earth would not be factored into the position, but with "IAU_Earth" the location
///    would be relative to the current longitude and thus the object would spin with the Earth.
///
/// In addition, each CelestialAnchor has a transformation relative to the referenced SPICE
/// coordinate system. This transformation is given as a separate position, rotation, and scale.
///
/// This class also provides methods for getting the transformation components in the coordinate
/// system of other entities.
class CS_SCENE_EXPORT CelestialAnchor {
 public:
  explicit CelestialAnchor(
      std::string sCenterName = "Solar System Barycenter", std::string sFrameName = "J2000");

  CelestialAnchor(CelestialAnchor const& other) = default;
  CelestialAnchor(CelestialAnchor&& other)      = default;

  CelestialAnchor& operator=(CelestialAnchor const& other) = default;
  CelestialAnchor& operator=(CelestialAnchor&& other)      = default;

  virtual ~CelestialAnchor() = default;

  // -----------------------------------------------------------------------------------------------

  /// SPICE name of the center body.
  /// A reference frame’s center must be a SPICE ephemeris object whose location is coincident with
  /// the origin (0, 0, 0) of the frame.
  virtual std::string const& getCenterName() const;
  virtual void               setCenterName(std::string sCenterName);

  /// SPICE name of the frame.
  virtual std::string const& getFrameName() const;
  virtual void               setFrameName(std::string sFrameName);

  /// Additional translation in meters, relative to center in frame coordinates additional scaling
  /// and rotation is applied afterwards and will not change the position relative to the center.
  virtual glm::dvec3 const& getPosition() const;
  virtual void              setPosition(glm::dvec3 vPos);

  /// Additional rotation around the point center + position in frame coordinates.
  virtual glm::dquat const& getRotation() const;
  virtual void              setRotation(glm::dquat qRot);

  /// Additional uniform scaling around the point center + position.
  virtual double getScale() const;
  virtual void   setScale(double dScale);

  // -----------------------------------------------------------------------------------------------

  /// Returns the position of "other" in the coordinate system defined by this CelestialAnchor - the
  /// result is not affected by the additional rotation and scale of "other", as these do not change
  /// it's position. This may throw a std::runtime_error if no sufficient SPICE data is available.
  virtual glm::dvec3 getRelativePosition(double tTime, CelestialAnchor const& other) const;

  /// Returns the rotation which aligns the coordinate system of this CelestialAnchor with "other" -
  /// the calculation depends on both frames and additional rotations. This may throw a
  /// std::runtime_error if no sufficient SPICE data is available.
  virtual glm::dquat getRelativeRotation(double tTime, CelestialAnchor const& other) const;

  /// Returns the entire transformation of "other" in the coordinate system defined by this
  /// CelestialAnchor. This may throw a std::runtime_error if no sufficient SPICE data is available.
  virtual glm::dmat4 getRelativeTransform(double tTime, CelestialAnchor const& other) const;

  /// Returns the how much "other" is larger than this, i.e. other.GetAnchorScale() /
  /// GetAnchorScale().
  virtual double getRelativeScale(CelestialAnchor const& other) const;

 protected:
  glm::dvec3 mPosition;
  glm::dquat mRotation;
  double     mScale{1.0};

  std::string mCenterName;
  std::string mFrameName;
};

} // namespace cs::scene

#endif // CS_SCENE_CELESTIAL_ANCHOR_HPP
