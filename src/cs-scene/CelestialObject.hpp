////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_SCENE_CELESTIAL_OBJECT_HPP
#define CS_SCENE_CELESTIAL_OBJECT_HPP

#include "../cs-utils/Property.hpp"
#include "CelestialAnchor.hpp"

#include <limits>
#include <memory>

class VistaOpenGLNode;

namespace cs::scene {

/// CelestialObjects have a lifetime in the universe. They are defined by their start and end
/// existence time. The time is given in the Barycentric Dynamical Time format, which is used
/// throughout SPICE.
class CS_SCENE_EXPORT CelestialObject : public CelestialAnchor {
 public:
  /// This will be set during update() according to the given radii consider this to be read-only.
  utils::Property<bool> pVisible = false;

  /// Creates a new CelestialObject.
  ///
  /// @param sCenterName      The SPICE name of the object.
  /// @param sFrameName       The SPICE name of the reference frame.
  /// @param radii            These will be used for visibility culling. If set to glm::dvec3(0.0),
  ///                         pVisible will not change during update().
  /// @param tStartExistence  The point in Barycentric Dynamical Time in which the object started
  ///                         existing.
  /// @param tEndExistence    The point in Barycentric Dynamical Time in which the object ceased
  ///                         existing.
  CelestialObject(std::string const& sCenterName, std::string const& sFrameName,
      glm::dvec3 radii           = glm::dvec3(0.0),
      double     tStartExistence = std::numeric_limits<double>::lowest(),
      double     tEndExistence   = std::numeric_limits<double>::max());

  CelestialObject(CelestialObject const& other) = default;
  CelestialObject(CelestialObject&& other)      = default;

  CelestialObject& operator=(CelestialObject const& other) = default;
  CelestialObject& operator=(CelestialObject&& other) = default;

  virtual ~CelestialObject() = default;

  virtual glm::dmat4 const& getWorldTransform() const;
  virtual glm::dvec4        getWorldPosition() const;

  /// The time (in the Barycentric Dynamical Time format) at which the object starts to exist in
  /// the universe.
  double getStartExistence() const;
  void   setStartExistence(double value);

  /// The time (in the Barycentric Dynamical Time format) at which the object ceases to exist in
  /// the universe.
  double getEndExistence() const;
  void   setEndExistence(double value);

  /// The radii of the CelestialBody in meters. This will serve as a basis for visibility
  /// calculation if set to glm::dev3(0.0), pVisible will not change during update().
  glm::dvec3 const& getRadii() const;
  void              setRadii(glm::dvec3 const& value);

  /// This is called once a frame by the SolarSystem if this CelestialObject has been registered
  /// with the SolarSystem.
  void update(double tTime, CelestialObserver const& oObs) override;

  /// @return true, if the current time is in between the start and end existence values.
  virtual bool getIsInExistence() const;

 protected:
  glm::dmat4 matWorldTransform;
  glm::dvec3 mRadii;
  double     mStartExistence, mEndExistence;

  bool mIsInExistence = false;
};

} // namespace cs::scene

#endif // CS_SCENE_CELESTIAL_OBJECT_HPP
