////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_NODE_EDITOR_NODE_FACTORY_HPP
#define CSL_NODE_EDITOR_NODE_FACTORY_HPP

#include <functional>
#include <unordered_map>
#include <vector>

namespace csl::nodeeditor {

class NodeFactory {
 public:
  void registerSocketType(
      std::string const& name, std::string color, std::vector<std::string> compatibleTo = {});

  template <typename T, typename... Args>
  void registerNodeType(Args... args) {
    mNodeSourceFuncs.push_back([=]() { return T::getSource(); });
    mNodeCreateFuncs[T::getName()] = [=]() { T::create(args...); };
  }

  std::string getSocketSource() const;
  std::string getNodeSource() const;
  std::string getRegisterSource() const;

 private:
  struct SocketInfo {
    std::string              mColor;
    std::vector<std::string> mCompatibleTo;
  };

  std::unordered_map<std::string, SocketInfo>                mSockets;
  std::vector<std::function<std::string(void)>>              mNodeSourceFuncs;
  std::unordered_map<std::string, std::function<void(void)>> mNodeCreateFuncs;
};

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_NODE_FACTORY_HPP
