////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VIRTUAL_SATELLITE_REST_REQUEST_MANAGER_HPP
#define CSP_VIRTUAL_SATELLITE_REST_REQUEST_MANAGER_HPP

namespace csp::virtualsatellite {

class RestRequestManager {
 public:
  RestRequestManager() = default;

  explicit RestRequestManager(
      std::string basePath, std::string username = "", std::string password = "");

  nlohmann::json getRequest(
      std::string const& path, std::map<std::string, std::string> const& queryParams = {}) const;

  void setUsername(std::string const& username);
  void setPassword(std::string const& password);

 private:
  std::string mBasePath;
  std::string mUsername;
  std::string mPassword;

  // Helper method to URL encode parameters
  static std::string urlEncode(const std::string& value);
};

} // namespace csp::virtualsatellite

#endif // CSP_VIRTUAL_SATELLITE_REST_REQUEST_MANAGER_HPP
