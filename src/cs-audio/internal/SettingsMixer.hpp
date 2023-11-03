////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CS_AUDIO_SETTINGS_MIXER_HPP
#define CS_AUDIO_SETTINGS_MIXER_HPP

#include "cs_audio_export.hpp"

#include <memory>
#include <map>
#include <any>
#include <string>

namespace cs::audio {

/// @brief This static class acts as the tool kit for the UpdateConstructor. It provides all
/// functions needed to mix source settings for the pipeline. 
class CS_AUDIO_EXPORT SettingsMixer {
 public:
  /// @brief Modifies A. Deletes all keys in A that are also a key in B.
  /// @param A source settings
  /// @param B source settings
  static void A_Without_B(
    std::shared_ptr<std::map<std::string, std::any>> A, 
    std::shared_ptr<std::map<std::string, std::any>> B);

  /// @brief Modifies A. Deletes all keys in A that are part of the list of B.
  /// @param A source settings
  /// @param B list of settings keys
  static void A_Without_B(
    std::shared_ptr<std::map<std::string, std::any>> A, 
    std::shared_ptr<std::vector<std::string>> B);

  /// @brief Modifies A. Deletes all elements in A which's value is the same as B.
  /// @param A source settings
  /// @param B value to remove
  static void A_Without_B_Value(
    std::shared_ptr<std::map<std::string, std::any>> A, 
    std::string B);

  /// @brief Modifies A. Adds all elements of B to A or, if the key already exists, overrides them.
  /// @param A source settings
  /// @param B source settings
  static void OverrideAdd_A_with_B(
    std::shared_ptr<std::map<std::string, std::any>> A, 
    std::shared_ptr<std::map<std::string, std::any>> B);

  /// @brief Modifies A. Adds all elements of B to A if A is not already defining the same key.
  /// @param A source settings
  /// @param B source settings
  static void Add_A_with_B_if_not_defined(
    std::shared_ptr<std::map<std::string, std::any>> A, 
    std::shared_ptr<std::map<std::string, std::any>> B);
};

} // namespace cs::audio

#endif // CS_AUDIO_SETTINGS_MIXER_HPP
