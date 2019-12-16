////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../src/cs-utils/convert.hpp"
#include "../../src/cs-utils/doctest.hpp"

namespace cs::utils {

TEST_CASE("cs::utils::toDegrees") {
  CHECK(convert::toDegrees<float>(0.f) == 0.f);
  CHECK(convert::toDegrees<float>(M_PI_4) == doctest::Approx(45.f));
  CHECK(convert::toDegrees<float>(M_PI_2) == doctest::Approx(90.f));
  CHECK(convert::toDegrees<float>(M_PI) == doctest::Approx(180.f));

  // TODO Should that work like this?
  // CHECK(convert::toDegrees<float>(2.f * M_PI) == doctest::Approx(0.f));

  CHECK(convert::toDegrees<double>(0.0) == 0.0);
  CHECK(convert::toDegrees<double>(M_PI_4) == doctest::Approx(45.0));
  CHECK(convert::toDegrees<double>(M_PI_2) == doctest::Approx(90.0));
  CHECK(convert::toDegrees<double>(M_PI) == doctest::Approx(180.0));

  // TODO Should that work like this?
  // CHECK(convert::toDegrees<double>(2 * M_PI) == doctest::Approx(0.0));

  CHECK(convert::toDegrees<int32_t>(0) == 0);
  CHECK(convert::toDegrees<uint32_t>(0u) == 0u);
}

TEST_CASE("cs::utils::toRadians") {
  CHECK(convert::toRadians<float>(0.0f) == 0.0f);
  CHECK(convert::toRadians<float>(45.f) == doctest::Approx(M_PI_4));
  CHECK(convert::toRadians<float>(90.f) == doctest::Approx(M_PI_2));
  CHECK(convert::toRadians<float>(180.f) == doctest::Approx(M_PI));

  CHECK(convert::toRadians<double>(0.0) == 0.0);
  CHECK(convert::toRadians<double>(45.0) == doctest::Approx(M_PI_4));
  CHECK(convert::toRadians<double>(90.0) == doctest::Approx(M_PI_2));
  CHECK(convert::toRadians<double>(180.0) == doctest::Approx(M_PI));

  CHECK(convert::toRadians<int32_t>(0) == 0);
  CHECK(convert::toRadians<uint32_t>(0u) == 0u);
}
} // namespace cs::utils