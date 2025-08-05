////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef PARAMETRIC_MODES_HPP
#define PARAMETRIC_MODES_HPP

#include <string>
#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////
// These modes can be used to sample various parametric phase functions to CSV files.             //
////////////////////////////////////////////////////////////////////////////////////////////////////

int cornetteShanksMode(std::vector<std::string> const& arguments);
int henyeyGreensteinMode(std::vector<std::string> const& arguments);
int doubleHenyeyGreensteinMode(std::vector<std::string> const& arguments);

#endif // PARAMETRIC_MODES_HPP