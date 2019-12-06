# Configuring the IDE

## CLion
### Linux
Sadly you have to add the following lines to the `clion.sh` file, which is located in you CLion `bin` folder:
```
export LD_LIBRARY_PATH=../lib:../lib/DriverPlugins:$LD_LIBRARY_PATH
export VISTACORELIBS_DRIVER_PLUGIN_DIRS=../lib/DriverPlugins
```

- Run: ./make_externals.sh
- Run: ./make.sh
- Go to: _Settings_ -> _Build, Execution, Deployment_ -> _CMake_
- Release Profile
  - _Generation path_: `build/linux-release`
  - _Build options_: `--parallel 8`
  - _CMAKE options_:
```
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_INSTALL_PREFIX="<path to cosmoscout>/install/linux-release"
-DCOSMOSCOUT_EXTERNALS_DIR="<path to cosmoscout>/install/linux-externals-release"
-DCMAKE_EXPORT_COMPILE_COMMANDS=On
```

- Debug Profile
  - _Generation path_: `build/linux-debug`
  - _Build options_: `--parallel 8`
  - _CMAKE options_:
```
-DCMAKE_BUILD_TYPE=Debug
-DCMAKE_INSTALL_PREFIX="<path to cosmoscout>/install/linux-debug"
-DCOSMOSCOUT_EXTERNALS_DIR="<path to cosmoscout>/install/linux-externals-release"
-DCMAKE_EXPORT_COMPILE_COMMANDS=On
```

- Go to: _Run/Debug Configuration_ -> _CMake Application_ -> _cosmoscout_
- Release profile
  - _Target_: `cosmoscout`
  - _Executable_ -> _Select other_: `<path to cosmoscout>/install/linux-release/bin/cosmoscout`
  - _Program arguments_: `--settings=../share/config/simple_desktop.json -vista vista.ini`
  - _Working directory_: `<path to cosmoscout>/install/linux-release/bin`
  - _Before launch_ -> _+_ -> _Install_
- Debug profile
  - _Target_: `cosmoscout`
  - _Executable_ -> _Select other_: `<path to cosmoscout>/install/linux-debug/bin/cosmoscout`
  - _Program arguments_: `--settings=../share/config/simple_desktop.json -vista vista.ini`
  - _Working directory_: `<path to cosmoscout>/install/linux-debug/bin`
  - _Before launch_ -> _+_ -> _Install_

### Windows
_TODO_

### Recommended Plugins
- [.gitignore](https://plugins.jetbrains.com/plugin/7495--ignore/)
- [Awesome Console](https://plugins.jetbrains.com/plugin/7677-awesome-console/)
- [CodeGlance](https://plugins.jetbrains.com/plugin/7275-codeglance/)
- [GitToolBox](https://plugins.jetbrains.com/plugin/7499-gittoolbox/)
- [GLSL Support](https://plugins.jetbrains.com/plugin/6993-glsl-support/)
