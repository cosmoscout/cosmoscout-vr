////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_LONG_LAT_NODE_HPP
#define CSP_VISUAL_QUERY_LONG_LAT_NODE_HPP

#include "../../../../csl-node-editor/src/Node.hpp"
#include "../../../../csl-tools/src/Mark.hpp"

namespace csp::visualquery {

/// The LongLat node will provide a draggable marker on the surface of a celestial body. The
/// latitude and longitude of the marker will be provided as output.
class LongLat : public csl::nodeeditor::Node {
 public:
  // static interface ------------------------------------------------------------------------------

  static const std::string        sName;
  static std::string              sSource();
  static std::unique_ptr<LongLat> sCreate(std::shared_ptr<cs::core::InputManager> inputManager,
      std::shared_ptr<cs::core::SolarSystem>                                      solarSystem,
      std::shared_ptr<cs::core::Settings>                                         settings);

  // instance interface ----------------------------------------------------------------------------

  /// Constructor. The node will be created by the NodeEditor and passed to the constructor.
  LongLat(std::shared_ptr<cs::core::InputManager> inputManager,
      std::shared_ptr<cs::core::SolarSystem>      solarSystem,
      std::shared_ptr<cs::core::Settings>         settings);

  /// Destructor. The node will be destroyed by the NodeEditor.
  ~LongLat() override;

  /// Each node must override this. It simply returns the static sName.
  std::string const& getName() const override;

  /// Whenever the user changes the number value of the node, the LongLat will send a message to
  /// the C++ instance of the node via onMessageFromJS, which in turn will call the process()
  /// method. This simply updates the value of the 'value' output. This method may also get called
  /// occasionally by the node editor, for example if a new web client was connected hence needs
  /// updated values for all nodes.
  void process() override;

  /// This will be called every frame.
  void tick() override;

  /// This will be called whenever the CosmoScout.sendMessageToCPP() is called by the JavaScript
  /// client part of this node.
  /// @param message  A JSON object as sent by the JavaScript node. In this case, it is actually
  ///                 just the currently selected value.
  void onMessageFromJS(nlohmann::json const& message) override;

  /// This is called whenever the node needs to be serialized. It returns a JSON object containing
  /// the current value.
  nlohmann::json getData() const override;

  /// This is called whenever the node needs to be deserialized. The given JSON object should
  /// contain a number under the key "value". the current value.
  void setData(nlohmann::json const& json) override;

 private:
  std::shared_ptr<cs::core::InputManager> mInputManager;
  std::shared_ptr<cs::core::SolarSystem>  mSolarSystem;
  std::shared_ptr<cs::core::Settings>     mSettings;

  int  mOnClickConnection = -1;
  bool mWaitingForClick   = false;

  std::pair<double, double>         mValue;
  std::unique_ptr<csl::tools::Mark> mMark;
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_LONG_LAT_NODE_HPP
