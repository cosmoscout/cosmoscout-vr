////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../src/cs-utils/utils.hpp"
#include "../../src/cs-utils/doctest.hpp"

namespace cs::utils {
TEST_CASE("cs::utils::contains(std::string, std::string)") {
  CHECK_UNARY(contains("", ""));
  CHECK_UNARY(contains("a", ""));
  CHECK_UNARY(contains("ab", "a"));
  CHECK_UNARY(contains("aa", "a"));

  CHECK_UNARY_FALSE(contains("", "a"));
  CHECK_UNARY_FALSE(contains("b", "a"));
  CHECK_UNARY_FALSE(contains("ab", "ba"));
  CHECK_UNARY_FALSE(contains("a", " "));
};
} // namespace cs::utils