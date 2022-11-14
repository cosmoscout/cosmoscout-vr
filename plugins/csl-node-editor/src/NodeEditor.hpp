////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_NODE_EDITOR_NODE_EDITOR_HPP
#define CSL_NODE_EDITOR_NODE_EDITOR_HPP

#include "NodeFactory.hpp"

#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

class CivetServer;
class CivetHandler;
class CivetWebSocketHandler;

namespace csl::nodeeditor {

class CommunicationChannel;

/// The node editor creates a web server which serves a web frontend on a given port via HTTP. The
/// user can access this frontend with a web browser and start creating a node graph. For each
/// created node or connection, a C++ counterpart is instantiated on the backend. Any data flow
/// happens on the C++ side, the HTML / JavaScript graph is "just" a visualization of the graph.
/// Whenever a node in the graph needs to display some data, a message needs to be sent from the C++
/// backend to the JavaScript frontend. Similarly, whenever the user modifies the graph on the
/// frontend, a message is sent to the C++ backend. Custom node types have to inherit from the Node
/// class. This base class provides methods for communicating with the JavaScript counterpart of the
/// node. Internally, the communication happens via a web socket.
/// At any given time, there can be only one open connection to a client. If an additional client
/// connects, an error message will be shown instead of the frontend web page.
class CSL_NODE_EDITOR_EXPORT NodeEditor {
 public:
  /// This creates a new node editor instance. It will launch the web server in the background.
  /// @param port     The port on which the node editor serves the web frontend.
  /// @param factory  Use this to register node and socket types before creating the node editor.
  NodeEditor(uint16_t port, NodeFactory factory);
  ~NodeEditor();

  /// This needs to be called once each frame. If any node produced new data since the last call to
  /// update(), this will trigger a reprocessing of all necessary nodes.
  void update();

  /// This serializes the current node graph into a JSON structure which can later be used to
  /// restore the graph layout. The JSON format follows this structure:
  ///
  /// {
  ///     "nodes": {
  ///         <node ID>: {
  ///             "name": <node name>
  ///             "id": <node ID>
  ///             "position": [<x>, <y>],
  ///             "collapsed": <bool>
  ///             "data": { <a custom data object defined by the node type> }
  ///             "outputs" : {
  ///                 <from socket name> : {
  ///                     "connections": [
  ///                         {
  ///                             "node": <to node ID>,
  ///                             "input: <to socket name>
  ///                         },
  ///                         ...
  ///                     ]
  ///                 },
  ///                 ...
  ///             }
  ///         },
  ///         ...
  ///     }
  /// }
  /// @return A JSON representation of the current graph.
  nlohmann::json toJSON() const;

  /// This will replace the current graph with the graph in the given JSON object.
  /// @param json The JSON object must follow the structure defined above.
  /// @throws     The method will throw a std::runtime_error if the given JSON object does not match
  ///             the expected structure. If this happens, the graph may have been loaded only
  ///             partially.
  void fromJSON(nlohmann::json const& json);

 private:
  NodeFactory                                                        mFactory;
  std::shared_ptr<CommunicationChannel>                              mSocket;
  std::shared_ptr<NodeGraph>                                         mGraph;
  std::unique_ptr<CivetServer>                                       mServer;
  std::vector<std::pair<std::string, std::unique_ptr<CivetHandler>>> mHandlers;
};

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_NODE_EDITOR_HPP
