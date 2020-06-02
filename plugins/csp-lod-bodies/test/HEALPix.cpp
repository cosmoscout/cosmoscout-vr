////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../src/HEALPix.hpp"
#include "../../../src/cs-utils/doctest.hpp"

namespace csp::lodbodies {
TEST_CASE("csp::lodbodies::HEALPIX") {
  CHECK_EQ(HEALPix::getLevel(1).getLevel(), 1);
}
} // namespace csp::lodbodies