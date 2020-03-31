////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "GuiArea.hpp"
#include "GuiItem.hpp"
#include <algorithm>

#if defined(_WIN32) && defined(min)
#undef min
#endif

namespace cs::gui {

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiArea::addItem(GuiItem* item, unsigned int index) {
  index = std::min(index, static_cast<unsigned int>(mItems.size()));
  mItems.insert(mItems.begin() + index, item);
  item->onAreaResize(getWidth(), getHeight());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiArea::removeItem(GuiItem* item) {
  for (auto it(mItems.begin()); it != mItems.end(); ++it) {
    if (*it == item) {
      mItems.erase(it);
      return;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiArea::removeItem(unsigned int index) {
  removeItem(getItem(index));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GuiItem* GuiArea::getItem(unsigned int index) {
  if (index >= mItems.size()) {
    return nullptr;
  }

  return mItems.at(index);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

GuiItem* GuiArea::getItemAt(
    int areaX, int areaY, bool checkAlpha, bool excludeNoninteractive, bool excludeDisabled) {
  for (auto const& item : mItems) {
    if ((item->getIsInteractive() || !excludeNoninteractive) &&
        (item->getIsEnabled() || !excludeDisabled)) {
      int x{};
      int y{};
      if (item->calculateMousePosition(areaX, areaY, x, y)) {
        if (!checkAlpha || item->getAlpha(x, y) > 0) {
          return item;
        }
      }
    }
  }
  return nullptr;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<GuiItem*> const& GuiArea::getItems() const {
  return mItems;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GuiArea::updateItems() {
  for (auto&& item : mItems) {
    item->onAreaResize(getWidth(), getHeight());
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::gui
