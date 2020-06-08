////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_MINMAXPYRAMID_HPP
#define CSP_LOD_BODIES_MINMAXPYRAMID_HPP

#include <limits>
#include <vector>

namespace csp::lodbodies {

template <typename T>
class Tile;

/// The MinMaxPyramid is a data structure for finding lod data in constant time. It's similar
/// to a quad tree but it contains precomputed min and max values at each level.
class MinMaxPyramid {

 public:
  MinMaxPyramid();
  explicit MinMaxPyramid(Tile<float>* tile);

  MinMaxPyramid(MinMaxPyramid const& other) = default;
  MinMaxPyramid(MinMaxPyramid&& other)      = default;

  MinMaxPyramid& operator=(MinMaxPyramid const& other) = default;
  MinMaxPyramid& operator=(MinMaxPyramid&& other) = default;

  virtual ~MinMaxPyramid();

  std::vector<std::vector<float>>& getMinPyramid();
  std::vector<std::vector<float>>& getMaxPyramid();

  /// Returns the minimum value in the given quadrant.
  ///
  /// Requires a list of quadrant indices {[0..3], [0..3], [0..3], ...}
  /// The number of given quadrants (q == 3 -> q0: 4x4, q1: 2x2, q2: 1x1)
  /// determines the MinMaxPyramid level and the address for the data access.
  /// @code
  ///  e.g. [1, 1, 1] = 8
  ///  e.g. [1, 1, 2] = 15
  ///  e.g. [0, 1, 1] = 4
  ///  e.g. [0, 1, 2] = 11
  ///
  ///  1   2   3  |4|  5   6   7  |8|
  ///  9  10 |11| 12  13  14 |15| 16
  /// 17  18  19  20  21  22  23  24
  /// 25  26  27  28  29  30  31  32
  /// 33  34  35  36  37  38  39  40
  /// 41  42  43  44  45  46  47  48
  /// 49  50  51  52  53  54  55  56
  /// 57  58  59  60  61  62  63  64
  /// @endcode
  float getMin(std::vector<int> const& quadrants);
  float getMin() const {
    return mMinValue;
  }

  /// Returns the maximum value in the given quadrant.
  float getMax(std::vector<int> const& quadrants);
  float getMax() const {
    return mMaxValue;
  }

  /// The average value of the whole pyramid.
  float getAverage() const {
    return mAvgValue;
  }

 protected:
  static float getData(std::vector<std::vector<float>>& pyramid, std::vector<int> const& quadrants);

 private:
  std::vector<std::vector<float>> mMinPyramid;
  std::vector<std::vector<float>> mMaxPyramid;

  float mMinValue = std::numeric_limits<float>::max();
  float mMaxValue = std::numeric_limits<float>::lowest();
  float mAvgValue = 0.F;
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_MINMAXPYRAMID_HPP
