////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "RestRequestManager.hpp"
#include "logger.hpp"

#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>
#include <curlpp/cURLpp.hpp>

namespace csp::virtualsatellite {

RestRequestManager::RestRequestManager(
    std::string basePath, std::string username, std::string password)
    : mBasePath(std::move(basePath))
    , mUsername(std::move(username))
    , mPassword(std::move(password)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json RestRequestManager::getRequest(
    std::string const& path, std::map<std::string, std::string> const& queryParams) const {
  try {
    curlpp::Cleanup cleaner;
    curlpp::Easy    request;

    // Construct full URL with query parameters
    std::string fullUrl = mBasePath + path;

    if (!queryParams.empty()) {
      fullUrl += "?";
      std::vector<std::string> encodedParams;

      for (const auto& [key, value] : queryParams) {
        encodedParams.push_back(urlEncode(key) + "=" + urlEncode(value));
      }

      // Join parameters with &
      std::string result;
      for (size_t i = 0; i < encodedParams.size(); ++i) {
        if (i > 0)
          result += "&";
        result += encodedParams[i];
      }
      fullUrl += result;
    }

    request.setOpt<curlpp::options::Url>(fullUrl);
    request.setOpt<curlpp::options::HttpGet>(true);

    // Set authentication if credentials provided
    if (!mUsername.empty() && !mPassword.empty()) {
      request.setOpt<curlpp::options::UserPwd>(mUsername + ":" + mPassword);
    }

    // Add headers
    std::list<std::string> headers;
    headers.emplace_back("Accept: application/json");
    request.setOpt<curlpp::options::HttpHeader>(headers);

    // Prepare response stream
    std::ostringstream responseStream;
    request.setOpt<curlpp::options::WriteStream>(&responseStream);

    request.perform();

    return nlohmann::json::parse(responseStream.str());

  } catch (curlpp::RuntimeError& e) {
    logger().error("Runtime error: {}", e.what());
    return {};
  } catch (curlpp::LogicError& e) {
    logger().error("Logic error: {}", e.what());
    return {};
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void RestRequestManager::setUsername(std::string const& username) {
  mUsername = username;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void RestRequestManager::setPassword(std::string const& password) {
  mPassword = password;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string RestRequestManager::urlEncode(const std::string& value) {
  std::ostringstream escaped;
  escaped.fill('0');
  escaped << std::hex;

  for (const char c : value) {
    // Keep alphanumeric and safe characters
    if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
      escaped << c;
      continue;
    }

    // Encode special characters
    escaped << '%' << std::setw(2) << static_cast<int>(static_cast<unsigned char>(c));
  }

  return escaped.str();
}

} // namespace csp::virtualsatellite