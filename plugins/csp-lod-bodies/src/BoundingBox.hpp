////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_BOUNDINGBOX_HPP
#define CSP_LOD_BODIES_BOUNDINGBOX_HPP

#include <glm/glm.hpp>

namespace csp::lodbodies {

/// An axis aligned bounding box to calculate intersection. It is defined by the minimum and
/// maximum points.
template <typename FloatT>
class BoundingBox {
 public:
  using scalar_type = FloatT;

  BoundingBox();
  BoundingBox(glm::tvec3<FloatT> const& bbMin, glm::tvec3<FloatT> const& bbMax);

  glm::tvec3<FloatT> const& getMin() const;
  void                      setMin(glm::tvec3<FloatT> const& p);
  glm::tvec3<FloatT> const& getMax() const;
  void                      setMax(glm::tvec3<FloatT> const& p);

  // TODO wouldn't it be better two split this calculation into two methods?
  /// Calculates the intersection between a ray or line segment with the BoundingBox.
  ///
  /// @param      origin           The origin of the ray or the start of the line segment.
  /// @param      direction        The direction of the ray or the end of the line segment.
  /// @param      isRay            Determines if the intersection shall be calculated with a ray or
  ///                              line segment.
  /// @param[out] fMinIntersection The distance on the line or ray from the start to the begin of
  ///                              the intersection. DocTODO I think?
  /// @param[out] fMaxIntersection The distance on the line or ray from the start to the end of
  ///                              the intersection. DocTODO I think?
  /// @param      epsilon          The error allowed for floating point comparisons.
  ///
  /// @return If an intersection occurred at all.
  bool GetIntersectionDistance(glm::tvec3<FloatT> origin, glm::tvec3<FloatT> direction, bool isRay,
      FloatT& fMinIntersection, FloatT& fMaxIntersection, FloatT epsilon = 0.00001);

 private:
  glm::tvec3<FloatT> mMin;
  glm::tvec3<FloatT> mMax;
};

template <typename FloatT>
BoundingBox<FloatT>::BoundingBox()
    : mMin()
    , mMax() {
}

template <typename FloatT>
BoundingBox<FloatT>::BoundingBox(glm::tvec3<FloatT> const& bbMin, glm::tvec3<FloatT> const& bbMax)
    : mMin(bbMin)
    , mMax(bbMax) {
}

template <typename FloatT>
typename glm::tvec3<FloatT> const& BoundingBox<FloatT>::getMin() const {
  return mMin;
}

template <typename FloatT>
void BoundingBox<FloatT>::setMin(glm::tvec3<FloatT> const& p) {
  mMin = p;
}

template <typename FloatT>
typename glm::tvec3<FloatT> const& BoundingBox<FloatT>::getMax() const {
  return mMax;
}

template <typename FloatT>
void BoundingBox<FloatT>::setMax(glm::tvec3<FloatT> const& p) {
  mMax = p;
}

template <typename FloatT>
bool BoundingBox<FloatT>::GetIntersectionDistance(glm::tvec3<FloatT> origin,
    glm::tvec3<FloatT> direction, bool isRay, FloatT& fMinIntersection, FloatT& fMaxIntersection,
    FloatT epsilon) {
  FloatT directionNorm[3];
  directionNorm[0] = direction[0];
  directionNorm[1] = direction[1];
  directionNorm[2] = direction[2];

  // If the computation is done for a ray, normalize the
  // direction vector first.
  if (isRay) {
    const FloatT sqrlen =
        (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
    if (sqrlen > 0) {
      const FloatT inverse_length = 1.0 / (FloatT)sqrt(sqrlen);
      directionNorm[0] *= inverse_length;
      directionNorm[1] *= inverse_length;
      directionNorm[2] *= inverse_length;
    }
  }

  FloatT tmin = 0;
  FloatT tmax = 0;
  bool   init = true;
  // Check all three axis one by one.
  for (int i = 0; i < 3; ++i) {
    if (std::abs(directionNorm[i]) > epsilon) {
      // Compute the parametric values for the intersection
      // points of the line and the bounding box according
      // to the current axis only.
      FloatT tmpmin = (mMin[i] - origin[i]) / directionNorm[i];
      FloatT tmpmax = (mMax[i] - origin[i]) / directionNorm[i];

      if (tmpmin > tmpmax) {
        // Switch tmpmin and tmpmax.
        const FloatT tmp = tmpmin;
        tmpmin           = tmpmax;
        tmpmax           = tmp;
      }
      if (init) {
        tmin = tmpmin;
        tmax = tmpmax;

        if (tmax < -epsilon) {
          return false;
        }
        if (tmin < 0) {
          tmin = 0;
        }

        if (!isRay) { // is a line segment
          // First intersection is outside the scope of
          // the line segment.
          if (tmin > 1 + epsilon) {
            return false;
          }
          if (tmax > 1) {
            tmax = 1;
          }
        }

        init = false;
      } else {
        // This is the regular check if the direction
        // vector is non-zero along the current axis.
        if (tmpmin > tmax + epsilon) {
          return false;
        }
        if (tmpmax < tmin - epsilon) {
          return false;
        }
        if (tmpmin > tmin) {
          tmin = tmpmin;
        }
        if (tmpmax < tmax) {
          tmax = tmpmax;
        }
      }
    } else { // line parallel to box
      // If the ray or line segment is parallel to an axis
      // and has its origin outside the box's std::min and std::max
      // coordinate for that axis, the ray/line cannot hit
      // the box.
      if ((origin[i] < mMin[i] - epsilon) || (origin[i] > mMax[i] + epsilon)) {
        return false;
      }
    }
  }
  fMinIntersection = tmin;
  fMaxIntersection = tmax;
  return true;
}

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_BOUNDINGBOX_HPP
