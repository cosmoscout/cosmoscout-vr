////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "x11utils.hpp"

#include "logger.hpp"

#ifdef HAVE_X11
#include <GL/glx.h>
#include <X11/Xatom.h>
#include <X11/Xutil.h>
#include <stb_image.h>
#include <stb_image_resize.h>
#endif

namespace x11utils {

////////////////////////////////////////////////////////////////////////////////////////////////////

void setXClassHint(std::string const& title) {
#ifdef HAVE_X11
  // Setting both, res_name and res_class of the XClassHint structure seems to make it appear in
  // Unity's dash and Gnome Shell's Alt-Tab switcher.
  XClassHint hints;

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  hints.res_name = const_cast<char*>(title.data());

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  hints.res_class = const_cast<char*>(title.data());

  // Freeglut also does not give access to the native XWindow and XDisplay handles. We use GLX
  // here and hope for the best...
  auto* xDisplay = glXGetCurrentDisplay();
  auto  xWindow  = glXGetCurrentDrawable();
  XSetClassHint(xDisplay, xWindow, &hints);
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void setAppIcon(std::string const& file) {
#ifdef HAVE_X11
  // Freeglut also does not give access to the native XWindow and XDisplay handles. We use GLX
  // here and hope for the best...
  auto* xDisplay = glXGetCurrentDisplay();
  auto  xWindow  = glXGetCurrentDrawable();

  int width{};
  int height{};
  int bpp{};
  int channels = 4;

  // Load the icon.
  uint8_t* data = stbi_load(file.c_str(), &width, &height, &bpp, channels);

  if (data) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    std::vector<uint8_t> pixels(data, data + width * height * 4);
    stbi_image_free(data);

    const int downSampledIcons = 4;
    for (int i = 0; i < downSampledIcons; ++i) {

      // XChangeProperty requires the data to be packed as ARGB into 64 bit words. The first two
      // words are the width and the height.
      std::vector<uint64_t> encoded;
      encoded.resize(2 + width * height);

      encoded[0] = width;
      encoded[1] = height;

      for (int i = 0; i < width * height; ++i) {
        encoded[i + 2] = pixels[i * 4 + 3] << 24 | pixels[i * 4 + 0] << 16 |
                         pixels[i * 4 + 1] << 8 | pixels[i * 4 + 2];
      }

      // Now set the X11 property.
      XChangeProperty(xDisplay, xWindow, XInternAtom(xDisplay, "_NET_WM_ICON", false), XA_CARDINAL,
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
          32, PropModeAppend, reinterpret_cast<const unsigned char*>(encoded.data()),
          encoded.size());

      // Create a down-sampled version of the icon for the next loop iteration.
      if (i + 1 < downSampledIcons) {
        int newWidth  = width / 2;
        int newHeight = height / 2;

        std::vector<uint8_t> downSampled(newWidth * newHeight * 4);

        stbir_resize_uint8(
            pixels.data(), width, height, 0, downSampled.data(), newWidth, newHeight, 0, channels);

        pixels = downSampled;
        width  = newWidth;
        height = newHeight;
      }
    }

  } else {
    logger().warn("Failed to set application icon! Icon '{}' not found!", file);
  }
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace x11utils
