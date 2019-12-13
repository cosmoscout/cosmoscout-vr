<p align="center"> 
  <img src ="img/banner-ide.jpg" />
</p>

# Configuring your IDE

Below you find some instructions on how to setup your preferred IDE for CosmoScout VR development.

## CLion

### Linux

Sadly you have to add the following lines to the `clion.sh` file, which is located in you CLion `bin` folder:

```bash
export LD_LIBRARY_PATH=../lib:../lib/DriverPlugins:$LD_LIBRARY_PATH
export VISTACORELIBS_DRIVER_PLUGIN_DIRS=../lib/DriverPlugins
```

- Run: `./make_externals.sh`
- Run: `./make.sh`
- Go to: _Settings_ -> _Build, Execution, Deployment_ -> _CMake_
- Release Profile
  - _Generation path_: `build/linux-release`
  - _Build options_: `--parallel 8`
  - _CMAKE options_:
    ```bash
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="<path to cosmoscout>/install/linux-release"
    -DCOSMOSCOUT_EXTERNALS_DIR="<path to cosmoscout>/install/linux-externals-release"
    -DCMAKE_EXPORT_COMPILE_COMMANDS=On
    ```

- Debug Profile
  - _Generation path_: `build/linux-debug`
  - _Build options_: `--parallel 8`
  - _CMAKE options_:
    ```bash
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

## Visual Studio Code

For Visual Studio Code, only the [C/C++ Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) is required.
However, we can also recommend the following extensions: [CMake](https://marketplace.visualstudio.com/items?itemName=twxs.cmake), [Markdown Preview Github Styling](https://marketplace.visualstudio.com/items?itemName=bierner.markdown-preview-github-styles) and [Code Spell Checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker).

In order to get started, you will need to create three files in a `.vscode` directory in the root of CosmoScout's source tree, namely `.vscode/tasks.json`, `.vscode/launch.json` and `.vscode/c_cpp_properties.json`.
We will discuss these files in the following.

### `.vscode/tasks.json`

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Make (Release)",
      "type": "shell",
      "command": "./make.sh",
      "options": {
        "cwd": "${workspaceFolder}"
      },
      "problemMatcher": [
        "$gcc"
      ],
      "windows": {
        "command": ".\\make.bat"
      }
    },
    {
      "label": "Make (Debug)",
      "type": "shell",
      "command": "./make.sh",
      "options": {
        "cwd": "${workspaceFolder}",
        "env": {
          "COSMOSCOUT_DEBUG_BUILD": "true"
        }
      },
      "problemMatcher": [
        "$gcc"
      ],
      "windows": {
        "command": ".\\make.bat"
      }
    },
    {
      "label": "Make Externals (Release)",
      "type": "shell",
      "command": "./make_externals.sh",
      "options": {
        "cwd": "${workspaceFolder}"
      },
      "problemMatcher": [
        "$gcc"
      ],
      "windows": {
        "command": ".\\make_externals.bat"
      }
    },
    {
      "label": "Make Externals (Debug)",
      "type": "shell",
      "command": "./make_externals.sh",
      "options": {
        "cwd": "${workspaceFolder}",
        "env": {
          "COSMOSCOUT_DEBUG_BUILD": "true"
        }
      },
      "problemMatcher": [
        "$gcc"
      ],
      "windows": {
        "command": ".\\make_externals.bat"
      }
    },
    {
      "label": "Run CosmoScout VR",
      "type": "shell",
      "command": "install/linux-release/bin/start.sh",
      "options": {
        "cwd": "${workspaceFolder}"
      },
      "problemMatcher": [],
      "windows": {
        "command": ".\\install/windows-release/bin/start.bat"
      }
    }
  ]
}
```

With this file in place, you can press `Ctrl+Shift+P` and select `Tasks: Run Task`. Now you can first select `Make Externals (Release)`, then `Make (Release)` and later `Run CosmoScout VR`.

### `.vscode/c_cpp_properties.json`

```json
{
    "configurations": [
        {
            "name": "Linux",
            "compileCommands": "${workspaceRoot}/build/linux-release/compile_commands.json",
            "browse": {
                "databaseFilename": "${workspaceRoot}/.vscode/browse-linux.VC.db"
            },
            "includePath": [
                "${workspaceRoot}/build/linux-release/**"
            ],
            "defines": [],
            "cStandard": "c11",
            "cppStandard": "c++17",
            "intelliSenseMode": "clang-x64"
        },
        {
            "name": "Win32",
            "browse": {
                "databaseFilename": "${workspaceRoot}/.vscode/browse-windows.VC.db"
            },
            "includePath": [
                "${workspaceRoot}/build/windows-release/**",
                "${workspaceRoot}/install/windows-externals-release/include"
            ],
            "defines": [],
            "cStandard": "c11",
            "cppStandard": "c++17"
        }
    ],
    "version": 4
}
```

This file configures Intellisense. On Linux, the CMake flag `-DCMAKE_EXPORT_COMPILE_COMMANDS=On` is used in `make.sh` to create a database which is required by Intellisense. With this file and the [C/C++ Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools), autocompletion and similar functionality should be working both on Windows and on Linux.

### `.vscode/launch.json`

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debugger (Linux)",
      "type": "cppdbg",
      "request": "launch",
      "program": "${workspaceFolder}/install/linux-debug/bin/cosmoscout",
      "args": [
        "--settings=../share/config/simple_desktop.json",
        "-vistaini vista.ini"
      ],
      "stopAtEntry": false,
      "cwd": "${workspaceFolder}/install/linux-debug/bin",
      "environment": [
        {
          "name": "VISTACORELIBS_DRIVER_PLUGIN_DIRS",
          "value": "../lib/DriverPlugins"
        },
        {
          "name": "LD_LIBRARY_PATH",
          "value": "../lib:../lib/DriverPlugins:${env:LD_LIBRARY_PATH}"
        }
      ],
      "externalConsole": false,
      "MIMode": "gdb",
      "setupCommands": [
        {
          "description": "Enable pretty-printing for gdb",
          "text": "-enable-pretty-printing",
          "ignoreFailures": true
        }
      ]
    },
    {
      "name": "Debugger (Windows)",
      "type": "cppvsdbg",
      "request": "launch",
      "program": "${workspaceFolder}/install/windows-debug/bin/cosmoscout",
      "args": [
        "--settings=../share/config/simple_desktop.json",
        "-vistaini vista.ini"
      ],
      "stopAtEntry": false,
      "cwd": "${workspaceFolder}/install/windows-debug/bin",
      "environment": [
        {
          "name": "VISTACORELIBS_DRIVER_PLUGIN_DIRS",
          "value": "..\\lib\\DriverPlugins"
        },
        {
          "name": "PATH",
          "value": "../lib;${env:PATH}"
        }
      ],
      "externalConsole": false
    }
  ]
}
```

Finally, when this files is created, you can use `F5` to launch the debugger on Windows and on Linux.
For this to work, CosmoScout VR and its dependencies have to be built in debug mode.

<p align="center">
  <a href="install.md">&lsaquo; Generic Build Instructions</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="README.md">&#8962; Help Index</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="using.md">Using CosmoScout VR &rsaquo;</a>
</p>

<p align="center"><img src ="img/hr.svg"/></p>