////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../src/cs-utils/doctest.hpp"
#include "../../src/cs-utils/utils.hpp"

namespace cs::utils {
TEST_CASE("cs::utils::contains(std::string, std::string)") {
  CHECK(contains("", ""));
  CHECK(contains("a", ""));
  CHECK(contains("ab", "a"));
  CHECK(contains("aa", "a"));

  CHECK(!contains("", "a"));
  CHECK(!contains("b", "a"));
  CHECK(!contains("ab", "ba"));
  CHECK(!contains("a", " "));
};
}