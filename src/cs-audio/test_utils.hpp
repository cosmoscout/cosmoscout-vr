
#ifndef TEST_UTILS
#define TEST_UTILS

#include <memory>
#include <map>
#include <any>
#include <vector>
#include <iostream>

void printMap(std::shared_ptr<std::map<std::string, std::any>> map) {
  for (auto const& [key, val] : (*map)) {
    std::cout << key << ": ";

    try {
      int x = std::any_cast<int>(val);
      std::cout << x << std::endl;           
      continue;
    } catch (std::bad_any_cast&) {
    }

    try {
      bool x = std::any_cast<bool>(val);
      std::cout << (x ? "true" : "false") << std::endl;
      continue;
    } catch (std::bad_any_cast&) {
    }

    try {
      float x = std::any_cast<float>(val);
      std::cout << x << std::endl;
      continue;
    } catch (std::bad_any_cast&) {
    }

    std::cout << std::endl;
  }
}

void printVec(std::shared_ptr<std::vector<std::string>> vec) {
  for (auto val : (*vec)) {
    std::cout << val << std::endl;
  }
}

#endif