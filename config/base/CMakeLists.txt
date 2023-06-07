# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

# install platform dependent files -----------------------------------------------------------------
if (WIN32)
  install(FILES "spice/simple-windows.txt"            DESTINATION "share/config/spice" RENAME "simple.txt")
  install(FILES "scripts/start.bat"                   DESTINATION "bin")
  install(FILES "scripts/hmd.bat"                     DESTINATION "bin")

  if (COSMOSCOUT_UNIT_TESTS)
    install(FILES "scripts/run_tests.bat" DESTINATION "bin")
  endif()
endif()

if (UNIX)
  install(FILES "spice/simple-linux.txt"            DESTINATION "share/config/spice" RENAME "simple.txt")
  install(FILES "scripts/start.sh"                  DESTINATION "bin"
          PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ)
  install(FILES "scripts/hmd.sh"                    DESTINATION "bin"
          PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ)
  
  if (COSMOSCOUT_UNIT_TESTS)
    install(FILES "scripts/run_tests.sh" DESTINATION "bin"
            PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ)
    install(FILES "scripts/run_graphical_tests.sh" DESTINATION "bin"
            PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ)
  endif()
endif()

# install files for all platforms ------------------------------------------------------------------
install(FILES "scene/simple_desktop.json" DESTINATION "share/config")
install(FILES "scene/simple_hmd.json"     DESTINATION "share/config")
install(DIRECTORY "vista"                 DESTINATION "share/config")
