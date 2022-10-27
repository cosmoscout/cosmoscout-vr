////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "NodeFactory.hpp"

#include "logger.hpp"

namespace csl::nodeeditor {

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeFactory::registerSocketType(
    std::string const& name, std::string color, std::vector<std::string> compatibleTo) {
  mSockets[name] = {std::move(color), std::move(compatibleTo)};
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string NodeFactory::getSocketSource() const {
  std::string source;

  for (auto const& s : mSockets) {
    source += fmt::format("CosmoScout.socketTypes['{0}'] = new Rete.Socket('{0}');\n", s.first);
  }

  for (auto const& s : mSockets) {
    source += fmt::format("addSocketStyle('{}', '{}');\n", s.first, s.second.mColor);
  }

  for (auto const& s : mSockets) {
    for (auto const& o : s.second.mCompatibleTo) {
      source += fmt::format(
          "CosmoScout.socketTypes['{}'].combineWith(CosmoScout.socketTypes['{}']);\n", s.first, o);
    }
  }

  return source;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string NodeFactory::getNodeSource() const {
  std::string source;

  for (auto const& f : mNodeSourceFuncs) {
    source += f();
  }

  return source;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string NodeFactory::getRegisterSource() const {
  std::string source;

  for (auto const& f : mNodeCreateFuncs) {
    source += "{\n";
    source += fmt::format("const component = new {}Component();\n", f.first);
    source += "CosmoScout.nodeEditor.register(component);\n";
    source += "}\n";
  }

  return source;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<Node> NodeFactory::createNode(std::string const& type) const {
  auto func = mNodeCreateFuncs.find(type);

  if (func == mNodeCreateFuncs.end()) {
    throw std::runtime_error(
        "Failed to create node of type '" + type + "': This type has not been registered!");
  }

  return func->second();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csl::nodeeditor
