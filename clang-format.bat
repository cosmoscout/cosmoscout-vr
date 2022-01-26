@echo off

rem ---------------------------------------------------------------------------------------------- #
rem                              This file is part of CosmoScout VR                                #
rem     and may be used under the terms of the MIT license. See the LICENSE file for details.      #
rem                       Copyright: (c) 2019 German Aerospace Center (DLR)                        #
rem ---------------------------------------------------------------------------------------------- #

@echo on
for /R %%f in (*.cpp) do clang-format -style file -i "%%f" -verbose
for /R %%f in (*.hpp) do clang-format -style file -i "%%f" -verbose
