////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "RequestHandler.hpp"

#include <include/wrapper/cef_stream_resource_handler.h>

#include <fstream>
#include <iostream>

namespace cs::gui::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////

CefRefPtr<CefResourceHandler> RequestHandler::GetResourceHandler(
    CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, CefRefPtr<CefRequest> request) {

  std::string url(request->GetURL().ToString());

  if (url.find("file://") == 0) {
    std::string path(url.substr(7));
    std::string ext(url.substr(url.find_last_of('.')));

    std::ifstream input(path, std::ios::binary);

    if (!input) {
      std::cout << "Failed to open file '" << path << "'!" << std::endl;
      return nullptr;
    }

    std::vector<char> buffer(
        (std::istreambuf_iterator<char>(input)), (std::istreambuf_iterator<char>()));

    CefRefPtr<CefStreamReader> stream =
        CefStreamReader::CreateForData(static_cast<void*>(buffer.data()), buffer.size());

    std::string mime("text/html");
    if (ext == ".png") {
      mime = "image/png";
    } else if (ext == ".jpg") {
      mime = "image/jpg";
    } else if (ext == ".jpeg") {
      mime = "image/jpg";
    } else if (ext == ".js") {
      mime = "text/javascript";
    } else if (ext == ".css") {
      mime = "text/css";
    } else if (ext == ".ttf") {
      mime = "application/x-font-ttf";
    } else if (ext == ".woff") {
      mime = "application/x-font-woff";
    } else if (ext == ".woff2") {
      mime = "application/x-font-woff";
    } else if (ext != ".html") {
      std::cout << "Opening file with unknown extension '" << ext << "'!" << std::endl;
    }

    return new CefStreamResourceHandler(mime, stream);
  }

  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool RequestHandler::OnCertificateError(CefRefPtr<CefBrowser> browser, cef_errorcode_t cert_error,
    CefString const& request_url, CefRefPtr<CefSSLInfo> ssl_info,
    CefRefPtr<CefRequestCallback> callback) {

  std::cout << "Detected certificate error. Continuing..." << std::endl;

  callback->Continue(true);
  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui::detail
