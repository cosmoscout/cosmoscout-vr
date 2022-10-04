////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

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
