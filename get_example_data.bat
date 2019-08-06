@echo off

rem ---------------------------------------------------------------------------------------------- #
rem                               This file is part of CosmoScout VR                               #
rem      and may be used under the terms of the MIT license. See the LICENSE file for details.     #
rem                         Copyright: (c) 2019 German Aerospace Center (DLR)                      #
rem ---------------------------------------------------------------------------------------------- #

rem create the data directory if necessary
set DATA_DIR=%~dp0\data

rem download the hipparcos and the tycho2 catalogue
mkdir "%DATA_DIR%\stars"
cd "%DATA_DIR%\stars"

powershell.exe -command Invoke-WebRequest -Uri ftp://ftp.imcce.fr/pub/catalogs/HIPP/cats/hip_main.dat -OutFile hip_main.dat
powershell.exe -command Invoke-WebRequest -Uri ftp://ftp.imcce.fr/pub/catalogs/TYCHO-2/catalog.dat -OutFile tyc2_main.dat


rem download some basic spice kernels
mkdir "%DATA_DIR%\spice"
cd "%DATA_DIR%\spice"

powershell.exe -command $AllProtocols = [System.Net.SecurityProtocolType]'Tls11,Tls12'; [System.Net.ServicePointManager]::SecurityProtocol = $AllProtocols;  Invoke-WebRequest -Uri https://naif.jpl.nasa.gov/pub/naif/cosmographia/kernels/spice/pck/pck00010.tpc -OutFile pck00010.tpc
powershell.exe -command $AllProtocols = [System.Net.SecurityProtocolType]'Tls11,Tls12'; [System.Net.ServicePointManager]::SecurityProtocol = $AllProtocols;  Invoke-WebRequest -Uri https://naif.jpl.nasa.gov/pub/naif/cosmographia/kernels/spice/lsk/naif0011.tls -OutFile naif0011.tls
powershell.exe -command $AllProtocols = [System.Net.SecurityProtocolType]'Tls11,Tls12'; [System.Net.ServicePointManager]::SecurityProtocol = $AllProtocols;  Invoke-WebRequest -Uri https://naif.jpl.nasa.gov/pub/naif/cosmographia/kernels/spice/spk/cg_1950_2050_v01.bsp -OutFile cg_1950_2050_v01.bsp