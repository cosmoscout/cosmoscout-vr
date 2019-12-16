////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../src/cs-utils/utils.hpp"
#include "../../src/cs-utils/doctest.hpp"

namespace cs::utils {
TEST_CASE("cs::utils::contains(std::string, std::string)") {
  CHECK_UNARY(contains("lorem ipsum", "lor"));
  CHECK_UNARY(contains("lorem ipsum", "psum"));
  CHECK_UNARY(contains("lorem ipsum", "m"));

  CHECK_UNARY_FALSE(contains("lorem ipsum", "emi"));
  CHECK_UNARY_FALSE(contains("lorem ipsum", "sum "));

  CHECK_UNARY(contains("", ""));
  CHECK_UNARY(contains("a", ""));
  CHECK_UNARY(contains("ab", "a"));
  CHECK_UNARY(contains("aa", "a"));

  CHECK_UNARY_FALSE(contains("", "a"));
  CHECK_UNARY_FALSE(contains("b", "a"));
  CHECK_UNARY_FALSE(contains("ab", "ba"));
  CHECK_UNARY_FALSE(contains("a", " "));
};

TEST_CASE("cs::utils::toString") {
  CHECK_EQ(toString(5), "5");
}

TEST_CASE("cs::utils::startsWith") {
  CHECK_UNARY(startsWith("lorem ipsum", "lor"));
  CHECK_UNARY(startsWith("lorem ipsum", "lorem ipsum"));

  CHECK_UNARY_FALSE(startsWith("lorem ipsum", "orem"));
  CHECK_UNARY_FALSE(startsWith("lorem ipsum", "lorem ipsum foo"));
  CHECK_UNARY_FALSE(startsWith("lorem ipsum", "abracadabra simsalabim"));
}

TEST_CASE("cs::utils::endsWith") {
  CHECK_UNARY(endsWith("lorem ipsum", "sum"));
  CHECK_UNARY(endsWith("lorem ipsum", "m"));
  CHECK_UNARY(endsWith("lorem ipsum", "lorem ipsum"));
  CHECK_UNARY_FALSE(endsWith("lorem ipsum", "ipsu"));
  CHECK_UNARY_FALSE(endsWith("lorem ipsum", "foo lorem ipsum"));
  CHECK_UNARY_FALSE(endsWith("lorem ipsum", "abracadabra simsalabim"));
}

} // namespace cs::utils