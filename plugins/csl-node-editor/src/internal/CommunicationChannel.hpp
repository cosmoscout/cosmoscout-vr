////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_NODE_EDITOR_COMMUNICATION_CHANNEL_HPP
#define CSL_NODE_EDITOR_COMMUNICATION_CHANNEL_HPP

#include <CivetServer.h>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <queue>

namespace csl::nodeeditor {

/// Communication between the C++ server and the JavaScript frontend of the node editor happens via
/// a web socket. There is a set of predefined event types which make up the whole communication.
/// This class is instantiated by the node editor. As a user of this library, you should not have to
/// use this class directly.
class CommunicationChannel : public CivetWebSocketHandler {

 public:
  struct Event {
    enum class Type {
      // General connection events -----------------------------------------------------------------

      /// This event is emitted by the CommunicationChannel whenever a new client has been connected
      /// successfully. For this event, the mData object is empty.
      eConnectionEstablished,

      /// This event is emitted by the CommunicationChannel whenever the current client has been
      /// disconnected. For this event, the mData object is empty.
      eConnectionDropped,

      // Events sent from C++ -> JavaScript --------------------------------------------------------

      // Whenever a new client connects, the node editor will send the entire graph to the new
      // client. For this event, the mData object contains the graph as returned by
      // NodeEditor::toJSON().
      eLoadGraph,

      // Events sent from JavaScript -> C++ --------------------------------------------------------

      /// Loading a graph on the frontend happens asynchronously. Once it is finished, the client
      /// will send this message. For this event, the mData object is empty.
      eGraphLoaded,

      /// Whenever the user adds a new node to the graph, this event is sent. The mData object is
      /// structured as follows:
      /// {
      ///   "type": <node type>,
      ///   "id": <node id>,
      ///   "position": [<x>, <y>]
      /// }
      eAddNode,

      /// Whenever the user removes a node from the graph, this event is sent. The mData object is
      /// structured as follows:
      /// {
      ///   "id": <node id>
      /// }
      eRemoveNode,

      /// Whenever the user moves a node around, this event is sent. The mData object is structured
      /// as follows:
      /// {
      ///   "id": <node id>,
      ///   "position": [<x>, <y>]
      /// }
      eTranslateNode,

      /// Whenever the user collapses or un-collapses a node to the graph, this event is sent. The
      /// mData object is structured as follows:
      /// {
      ///   "id": <node id>,
      ///   "collapsed": <bool>
      /// }
      eCollapseNode,

      /// Whenever the user adds a new node connection to the graph, this event is sent. The mData
      /// object is structured as follows:
      /// {
      ///   "fromNode": <node id>,
      ///   "fromSocket": <socket name>,
      ///   "toNode": <node id>,
      ///   "toSocket": <socket name>
      /// }
      eAddConnection,

      /// Whenever the user removes a node connection from the graph, this event is sent. The mData
      /// object is structured as follows:
      /// {
      ///   "fromNode": <node id>,
      ///   "fromSocket": <socket name>,
      ///   "toNode": <node id>,
      ///   "toSocket": <socket name>
      /// }
      eRemoveConnection,

      // Bidirectional events ----------------------------------------------------------------------

      /// Whenever a node sends a message to its JavaScript or C++ counterpart, this event is sent.
      /// The mData object contains the destination node ID as well as a custom message object. What
      /// this contains, depends on the node:
      /// {
      ///   "toNode": <node id>,
      ///   "message": { ... }
      /// }
      eNodeMessage
    };

    Type           mType;
    nlohmann::json mData;
  };

  /// Events are received in a separate thread and stored in a queue. This method can be used to get
  /// the received events one by one.
  /// @return The next event from the event queue.
  std::optional<Event> getNextEvent();

  /// Sends an event to the connected client (if any).
  /// @param event The event to send. The data object must contain the type-dependent fields as
  ///              described above.
  void sendEvent(Event const& event) const;

  /// Get whether there is a client connected currently.
  /// @return True if there is a client connected.
  bool isConnected() const;

 private:
  bool handleConnection(CivetServer* server, const struct mg_connection* conn) override;
  void handleReadyState(CivetServer* server, struct mg_connection* conn) override;
  bool handleData(CivetServer* server, struct mg_connection* conn, int bits, char* data,
      size_t data_len) override;
  void handleClose(CivetServer* server, const struct mg_connection* conn) override;

  std::queue<Event> mEventQueue;
  std::mutex        mEventQueueMutex;

  mg_connection* mConnection = nullptr;
};

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_COMMUNICATION_CHANNEL_HPP
