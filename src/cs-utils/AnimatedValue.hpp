////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_UTILS_ANIMATED_VALUE_HPP
#define CS_UTILS_ANIMATED_VALUE_HPP

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace cs::utils {

/// The values describe how an animation should play out.
enum class AnimationDirection {
  eIn,    ///< The beginning of the animation is slow, the end is fast.
  eOut,   ///< The beginning of the animation is fast, the end is slow.
  eInOut, ///< The beginning and end of the animation are slow, the middle of the animation is fast.
  eOutIn, ///< The beginning and end of the animation are fast, the middle of the animation is slow.
  eLinear ///< The animation has the same speed the whole time.
};
enum class AnimationLoop { eNone, eRepeat, eToggle };

/// A class for smooth value interpolation. It animates a value between a start and an end value,
/// given a start and an end time. It is also possible to define the way of interpolation. See
/// AnimationDirection for more details on that.
template <typename T>
class AnimatedValue {
 public:
  T                  mStartValue, mEndValue;
  double             mStartTime = 0.0;
  double             mEndTime   = 0.0;
  AnimationDirection mDirection = AnimationDirection::eInOut;
  double             mExponent  = 0.0;

  /// DocTODO why would anyone create an AnimatedValue, with the same value for start and end?
  explicit AnimatedValue(T const& val = T{})
      : mStartValue(val)
      , mEndValue(val) {
  }

  /// Creates a new AnimatedValue.
  /// @param startValue The value at the beginning of the animation.
  /// @param endValue   The value at the end of the animation.
  /// @param startTime  The time at which the animations result is equivalent to the startValue.
  /// @param endTime    The time at which the animations result is equivalent to the endValue.
  /// @param direction  The way the animation plays out. See AnimationDirection for more.
  /// @param exponent   How extreme the effects of non-linear directions are.
  AnimatedValue(T const& startValue, T const& endValue, double startTime, double endTime,
      AnimationDirection direction = AnimationDirection::eInOut, double exponent = 0.0)
      : mStartValue(startValue)
      , mEndValue(endValue)
      , mStartTime(startTime)
      , mEndTime(endTime)
      , mDirection(direction)
      , mExponent(exponent) {
  }

  /// @return Gives back an interpolated result according to the current settings and given time.
  T get(double time) {
    if (time < mStartTime) {
      return mStartValue;
    }

    if (time >= mEndTime) {
      return mEndValue;
    }

    double state = glm::clamp((time - mStartTime) / (mEndTime - mStartTime), 0.0, 1.0);

    switch (mDirection) {
    case AnimationDirection::eLinear:
      return updateLinear(state, mStartValue, mEndValue);
    case AnimationDirection::eIn:
      return updateEaseIn(state, mStartValue, mEndValue);
    case AnimationDirection::eOut:
      return updateEaseOut(state, mStartValue, mEndValue);
    case AnimationDirection::eInOut:
      return updateEaseInOut(state, mStartValue, mEndValue);
    default: // AnimationDirection::eOutIn:
      return updateEaseOutIn(state, mStartValue, mEndValue);
    }
  }

 protected:
  T updateLinear(double a, T const& s, T const& e) {
    return glm::mix(s, e, a);
  }

  T updateEaseIn(double a, T const& s, T const& e) {
    return glm::mix(s, e, (std::pow(a, 4.0) * ((mExponent + 1) * a - mExponent)));
  }

  T updateEaseOut(double a, T const& s, T const& e) {
    return glm::mix(s, e, (std::pow(a - 1, 4.0) * ((mExponent + 1) * (a - 1) + mExponent) + 1));
  }

  T updateEaseInOut(double a, T const& s, T const& e) {
    if (a < 0.5F) {
      return updateEaseIn(a * 2, s, glm::mix(s, e, 0.5));
    }

    return updateEaseOut(a * 2 - 1, glm::mix(s, e, 0.5), e);
  }

  T updateEaseOutIn(double a, T const& s, T const& e) {
    if (a < 0.5F) {
      return updateEaseOut(a * 2, s, glm::mix(s, e, 0.5));
    }

    return updateEaseIn(a * 2 - 1, glm::mix(s, e, 0.5), e);
  }
};

} // namespace cs::utils

#endif // CS_UTILS_ANIMATED_VALUE_HPP
