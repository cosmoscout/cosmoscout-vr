////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../src/cs-utils/Signal.hpp"
#include "../../src/cs-utils/doctest.hpp"

namespace cs::utils {
TEST_CASE("cs::utils:Signal::emit") {
  bool test = false;

  Signal<bool> a;
  a.connect([&](bool value) { test = value; });

  a.emit(true);
  CHECK(test);
  a.disconnectAll();
}

TEST_CASE("cs::utils:Signal::emitForAllButOne") {
  bool test1 = false;
  bool test2 = false;
  bool test3 = false;

  Signal<bool> a;

  int id = a.connect([&](bool value) { test1 = value; });
  a.connect([&](bool value) { test2 = value; });
  a.connect([&](bool value) { test3 = value; });

  a.emitForAllButOne(id, true);

  CHECK_FALSE(test1);
  CHECK(test2);
  CHECK(test3);

  a.disconnectAll();
}

TEST_CASE("cs::utils:Signal::emitFor") {
  bool test1 = false;
  bool test2 = false;
  bool test3 = false;

  Signal<bool> a;

  int id = a.connect([&](bool value) { test1 = value; });
  a.connect([&](bool value) { test2 = value; });
  a.connect([&](bool value) { test3 = value; });

  a.emitFor(id, true);

  CHECK(test1);
  CHECK_FALSE(test2);
  CHECK_FALSE(test3);

  a.disconnectAll();
}

TEST_CASE("cs::utils:Signal::disconnect") {
  bool test1 = false;
  bool test2 = false;

  Signal<bool> a;

  int id = a.connect([&](bool value) { test1 = value; });
  a.connect([&](bool value) { test2 = value; });

  a.emit(true);
  a.disconnect(id);
  a.emit(false);

  CHECK(test1);
  CHECK_FALSE(test2);

  a.disconnectAll();
}

TEST_CASE("cs::utils:Signal::disconnectAll") {
  bool test1 = false;
  bool test2 = false;

  Signal<bool> a;

  a.connect([&](bool value) { test1 = value; });
  a.connect([&](bool value) { test2 = value; });

  a.emit(true);
  a.disconnectAll();
  a.emit(false);

  CHECK(test1);
  CHECK(test2);
}

TEST_CASE("cs::utils::Signal disconnect one slot while emitting") {
  bool test1 = false;
  bool test2 = false;
  bool test3 = false;

  Signal<bool> a;

  a.connect([&](bool value) { test1 = value; });
  int id = a.connect([&](bool value) {
    test2 = value;
    a.disconnect(id);
  });
  a.connect([&](bool value) { test3 = value; });

  a.emit(true);

  CHECK(test1);
  CHECK(test2);
  CHECK(test3);

  a.emit(false);

  CHECK_FALSE(test1);
  CHECK(test2);
  CHECK_FALSE(test3);

  a.disconnectAll();
}

TEST_CASE("cs::utils::Signal disconnect one slot while emitting for all but one") {
  bool test1 = false;
  bool test2 = false;
  bool test3 = false;

  Signal<bool> a;

  int id1 = a.connect([&](bool value) { test1 = value; });
  int id2 = a.connect([&](bool value) {
    test2 = value;
    a.disconnect(id2);
  });
  int id3 = a.connect([&](bool value) { test3 = value; });

  a.emitForAllButOne(id1, true);

  CHECK_FALSE(test1);
  CHECK(test2);
  CHECK(test3);

  a.emitForAllButOne(id3, false);

  CHECK_FALSE(test1);
  CHECK(test2);
  CHECK(test3);

  a.disconnectAll();
}

TEST_CASE("cs::utils::Signal disconnect all slots while emitting") {
  bool test1 = false;
  bool test2 = false;
  bool test3 = false;

  Signal<bool> a;

  a.connect([&](bool value) { test1 = value; });
  a.connect([&](bool value) {
    test2 = value;
    a.disconnectAll();
  });
  a.connect([&](bool value) { test3 = value; });

  a.emit(true);

  CHECK(test1);
  CHECK(test2);
  CHECK(test3);

  a.emit(false);

  CHECK(test1);
  CHECK(test2);
  CHECK(test3);
}

TEST_CASE("cs::utils::Signal disconnect all slots while emitting for all but one") {
  bool test1 = false;
  bool test2 = false;
  bool test3 = false;

  Signal<bool> a;

  int id1 = a.connect([&](bool value) { test1 = value; });
  a.connect([&](bool value) {
    test2 = value;
    a.disconnectAll();
  });
  a.connect([&](bool value) { test3 = value; });

  a.emitForAllButOne(id1, true);

  CHECK_FALSE(test1);
  CHECK(test2);
  CHECK(test3);

  test1 = false;
  test2 = false;
  test3 = false;

  a.emit(true);

  CHECK_FALSE(test1);
  CHECK_FALSE(test2);
  CHECK_FALSE(test3);
}

} // namespace cs::utils