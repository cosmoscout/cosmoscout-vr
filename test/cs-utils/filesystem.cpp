////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "../../src/cs-utils/filesystem.hpp"
#include "../../src/cs-utils/doctest.hpp"

#include <iostream>

namespace cs::utils {

TEST_CASE("cs::utils::filesystem::create_and_list_dir") {
  cs::utils::filesystem::createDirectoryRecursively("./testDir/testDir");
  std::set<std::string> result = cs::utils::filesystem::listDirs("./testDir");

  CHECK_EQ(result.size(), 1);
  CHECK_EQ(*result.begin(), "./testDir/testDir");
};

TEST_CASE("cs::utils::filesystem::write_and_read") {
  cs::utils::filesystem::writeStringToFile("./testDir/testfile.txt", "test");
  std::string result = cs::utils::filesystem::loadToString("./testDir/testfile.txt");
  CHECK_EQ(result, "test");
};

TEST_CASE("cs::utils::filesystem::create_and_list_files_with_regex") {
  cs::utils::filesystem::writeStringToFile("./testDir/testfile.tmp", "test");
  std::set<std::string> result =
      cs::utils::filesystem::listFiles("./testDir", std::regex(".+\\.tmp"));

  CHECK_EQ(result.size(), 1);
  CHECK_EQ(*result.begin(), "./testDir/testfile.tmp");
};

TEST_CASE("cs::utils::filesystem::create_and_list_files") {
  std::set<std::string> result = cs::utils::filesystem::listFiles("./testDir");

  CHECK(result.size() == 2);
  CHECK_EQ(*result.begin(), "./testDir/testfile.tmp");
  CHECK_EQ(*(++result.begin()), "./testDir/testfile.txt");
};

} // namespace cs::utils