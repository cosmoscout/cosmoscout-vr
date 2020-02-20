////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ResourceRequestHandler.hpp"

#include <fstream>
#include <include/wrapper/cef_stream_resource_handler.h>
#include <iostream>
#include <spdlog/spdlog.h>

namespace cs::gui::detail {

////////////////////////////////////////////////////////////////////////////////////////////////////

CefRefPtr<CefResourceHandler> ResourceRequestHandler::GetResourceHandler(
    CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame, CefRefPtr<CefRequest> request) {

  std::string url(request->GetURL().ToString());

  if (url.find("file://") == 0) {
    std::string path(url.substr(7));
    std::string ext(url.substr(url.find_last_of('.')));

    std::ifstream input(path, std::ios::binary);

    if (!input) {
      spdlog::error("Failed to open gui resource: Cannot open file '{}'!", path);
      return nullptr;
    }

    std::vector<char> buffer(
        (std::istreambuf_iterator<char>(input)), (std::istreambuf_iterator<char>()));

    CefRefPtr<CefStreamReader> stream =
        CefStreamReader::CreateForData(static_cast<void*>(buffer.data()), buffer.size());

    std::string mime("text/html");
    if (ext == ".png") {
      mime = "image/png";
    } else if (ext == ".jpg" || ext == ".jpeg") {
      mime = "image/jpg";
    } else if (ext == ".js") {
      mime = "text/javascript";
    } else if (ext == ".css") {
      mime = "text/css";
    } else if (ext == ".ttf") {
      mime = "application/x-font-ttf";
    } else if (ext == ".woff" || ext == ".woff2") {
      mime = "application/x-font-woff";
    } else if (ext != ".html") {
      spdlog::warn("Opening file with unknown extension '{}'!", ext);
    }

    return new CefStreamResourceHandler(mime, stream);
  }

  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui::detail
