////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../src/cs-utils/convert.hpp"
#include "../../src/cs-utils/doctest.hpp"

namespace cs::utils {

const double PI   = 3.14159265359;
const double PI_2 = PI / 2.0;
const double PI_4 = PI / 4.0;

const float PI_F   = 3.14159265359F;
const float PI_2_F = PI_F / 2.0F;
const float PI_4_F = PI_F / 4.0F;

TEST_CASE("cs::utils::toDegrees") {
  CHECK_EQ(convert::toDegrees<float>(0.F), 0.F);
  CHECK_EQ(convert::toDegrees<float>(PI_4_F), doctest::Approx(45.F));
  CHECK_EQ(convert::toDegrees<float>(PI_2_F), doctest::Approx(90.F));
  CHECK_EQ(convert::toDegrees<float>(PI_F), doctest::Approx(180.F));

  CHECK_EQ(convert::toDegrees<double>(0.0), 0.0);
  CHECK_EQ(convert::toDegrees<double>(PI_4), doctest::Approx(45.0));
  CHECK_EQ(convert::toDegrees<double>(PI_2), doctest::Approx(90.0));
  CHECK_EQ(convert::toDegrees<double>(PI), doctest::Approx(180.0));

  CHECK_EQ(convert::toDegrees<int32_t>(0), 0);
  CHECK_EQ(convert::toDegrees<uint32_t>(0U), 0U);
}

TEST_CASE("cs::utils::toRadians") {
  CHECK_EQ(convert::toRadians<float>(0.0F), 0.0F);
  CHECK_EQ(convert::toRadians<float>(45.F), doctest::Approx(PI_4_F));
  CHECK_EQ(convert::toRadians<float>(90.F), doctest::Approx(PI_2_F));
  CHECK_EQ(convert::toRadians<float>(180.F), doctest::Approx(PI_F));

  CHECK_EQ(convert::toRadians<double>(0.0), 0.0);
  CHECK_EQ(convert::toRadians<double>(45.0), doctest::Approx(PI_4));
  CHECK_EQ(convert::toRadians<double>(90.0), doctest::Approx(PI_2));
  CHECK_EQ(convert::toRadians<double>(180.0), doctest::Approx(PI));

  CHECK_EQ(convert::toRadians<int32_t>(0), 0);
  CHECK_EQ(convert::toRadians<uint32_t>(0U), 0U);
}
} // namespace cs::utils