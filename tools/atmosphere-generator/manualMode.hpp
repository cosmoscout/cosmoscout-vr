////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef MANUAL_MODE_HPP
#define MANUAL_MODE_HPP

#include <string>
#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////
// This method allows writing CSV tables similar to the results from 'mie' or 'rayleigh' mode     //
// (e.g. scattering cross-sections, or absorption cross-sections) but with manually specified     //
// values.                                                                                        //
////////////////////////////////////////////////////////////////////////////////////////////////////

int manualMode(std::vector<std::string> const& arguments);

#endif // MANUAL_MODE_HPP