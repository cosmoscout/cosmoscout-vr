////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../../src/cs-utils/filesystem.hpp"
#include "../../src/cs-utils/doctest.hpp"

#include <iostream>

namespace cs::utils {

TEST_CASE("cs::utils::filesystem::create_and_list_dir") {
  cs::utils::filesystem::createDirectoryRecursively("./testDir/testDir");
  std::set<std::string> result = cs::utils::filesystem::listDirs("./testDir");

  CHECK(1 == result.size());

  std::set<std::string>::iterator it = result.begin();
  std::advance(it, 0);
  std::string entry = *it;

  CHECK_EQ("./testDir/testDir", entry);
};

TEST_CASE("cs::utils::filesystem::write_and_read") {
  cs::utils::filesystem::writeStringToFile("./testDir/testfile.txt","test");
  std::string result = cs::utils::filesystem::loadToString("./testDir/testfile.txt");
  CHECK_EQ("test", result);
};

TEST_CASE("cs::utils::filesystem::create_and_list_files_with_regex") {
  cs::utils::filesystem::writeStringToFile("./testDir/testfile.tmp","test");
  std::set<std::string> result = cs::utils::filesystem::listFiles("./testDir",std::regex(".+\\.tmp"));
 
  CHECK(1 == result.size());

  std::set<std::string>::iterator it = result.begin();
  std::advance(it, 0);
  std::string entry = *it;

  CHECK_EQ("./testDir/testfile.tmp", entry);
};

TEST_CASE("cs::utils::filesystem::create_and_list_files") {
  std::set<std::string> result = cs::utils::filesystem::listFiles("./testDir");
 
  CHECK(result.size() == 2);

  std::set<std::string>::iterator it = result.begin();
  std::advance(it, 0);
  std::string entry = *it;

  CHECK_EQ("./testDir/testfile.tmp", entry);

  std::advance(it, 1);
  entry = *it;

  CHECK_EQ("./testDir/testfile.txt", entry);
};


} // namespace cs::utils