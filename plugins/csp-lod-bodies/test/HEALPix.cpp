////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "../src/HEALPix.hpp"
#include "../../../src/cs-utils/doctest.hpp"

namespace csp::lodbodies {
TEST_CASE("csp::lodbodies::HEALPIX") {
  CHECK_EQ(HEALPix::getLevel(1).getLevel(), 1);
}
} // namespace csp::lodbodies