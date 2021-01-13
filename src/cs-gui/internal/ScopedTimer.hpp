////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GUI_SCOPED_TIMER_HPP
#define CS_GUI_SCOPED_TIMER_HPP

#include <string>

namespace cs::gui::detail {

/// Can be used to measure time taken by some part of code.
class ScopedTimer {

 public:
  explicit ScopedTimer(std::string name);
  ~ScopedTimer();

  ScopedTimer(ScopedTimer const& other) = default;
  ScopedTimer(ScopedTimer&& other)      = default;

  ScopedTimer& operator=(ScopedTimer const& other) = default;
  ScopedTimer& operator=(ScopedTimer&& other) = default;

 private:
  static double GetNow();

  std::string mName;
  double      mStartTime;
};

} // namespace cs::gui::detail

#endif // CS_GUI_SCOPED_TIMER_HPP
