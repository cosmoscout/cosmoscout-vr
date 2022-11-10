////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "NodeFactory.hpp"

#include "Node.hpp"
#include "logger.hpp"

namespace csl::nodeeditor {

////////////////////////////////////////////////////////////////////////////////////////////////////

// These need to be declared explicitely as the default versions would be defined inline in the
// header which makes it impossible to use a forward declartion of Node.
NodeFactory::NodeFactory()  = default;
NodeFactory::~NodeFactory() = default;

////////////////////////////////////////////////////////////////////////////////////////////////////

void NodeFactory::registerSocketType(std::string name, std::string color) {
  mSockets.emplace(std::move(name), std::move(color));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string NodeFactory::getSocketSource() const {
  std::string source;

  for (auto const& s : mSockets) {
    source += fmt::format("CosmoScout.socketTypes['{0}'] = new Rete.Socket('{0}');\n", s.first);
    source += fmt::format("addSocketStyle('{}', '{}');\n", s.first, s.second);
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