<!-- 
SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
SPDX-License-Identifier: CC-BY-4.0
 -->

<p align="center"> 
  <img src ="img/banner-ide.jpg" />
</p>

# Configuring your IDE

Below you find some instructions on how to setup your preferred IDE for CosmoScout VR development. 

* [CLion](#clion-linux--windows)
* [Visual Studio](#-visual-studio-windows-only)
* [Visual Studio Code](#-visual-studio-code-linux--windows)


## <img src="https://simpleicons.org/icons/clion.svg" alt="Simple Icons" width=24 height=18> CLion (Linux & Windows)

### Prerequisites

- Be sure to build the externals, as specified in the [installation](install.md) guide.
- On Windows ensure that `BOOST_ROOT` is registered as an environment variable. 

### Configure CMake Profiles

When opening the project you should be greeted with a CMake settings window. If you are not greeted with a CMake
settings window go to: *Settings -> Build, Execution, Deployment -> CMake*

You can choose to enable CMake presets. From the following you can choose one release and one debug config:

- Windows
  - windows-ninja-release-config - windows-ninja-release-build `preset`
  - windows-ninja-debug-config - windows-ninja-debug-build `preset`
  - windows-vs-release-config - windows-vs-release-build `preset`
  - windows-vs-debug-config - windows-vs-debug-build `preset`
- Linux
  - linux-ninja-release-config - linux-ninja-release-build `preset`
  - linux-ninja-debug-config - linux-ninja-debug-build `preset`
  - linux-make-release-config - linux-make-release-build `preset`
  - linux-make-debug-config - linux-make-debug-build `preset`

> [!IMPORTANT]
> 
> Be sure that you select presets that contain a config **AND** a build step.

> [!TIP]
> 
> By default CLion creates a profile called `Debug`. You can delete that.

> [!TIP]
> 
> You can create a copy of one of the presets and modify it to your likings.
  
### Configure Run and Test Profiles

- Run the `Install` task. It can be found under `Build` in the menubar.
- Go to: _Run/Debug Configuration_

#### Windows
- Run Release

  | **Type**                  | CMake Application                                                           |
  |---------------------------|-----------------------------------------------------------------------------|
  | **Target**                | `cosmoscout`                                                                |
  | **Executable**            | `$ProjectFileDir$\install\windows-Release\bin\cosmoscout.exe`               |
  | **Program arguments**     | `--settings=../share/config/simple_desktop.json -vista vista.ini`           |
  | **Environment variables** | `VISTACORELIBS_DRIVER_PLUGIN_DIRS=..\lib\DriverPlugins;PATH=..\lib\;$Path$` |
  | **Before launch**         | `install`                                                                   |

- Run Debug

  | **Type**                  | CMake Application                                                           |
  |---------------------------|-----------------------------------------------------------------------------|
  | **Target**                | `cosmoscout`                                                                |
  | **Executable**            | `$ProjectFileDir$\install\windows-Debug\bin\cosmoscout.exe`                 |
  | **Program arguments**     | `--settings=../share/config/simple_desktop.json -vista vista.ini`           |
  | **Environment variables** | `VISTACORELIBS_DRIVER_PLUGIN_DIRS=..\lib\DriverPlugins;PATH=..\lib\;$Path$` |
  | **Before launch**         | `install`                                                                   |

- Test Release

  | **Type**                  | Doctest                                           |
  |---------------------------|---------------------------------------------------|
  | **Target**                | `cosmoscout`                                      |
  | **Program arguments**     | `--run-tests --test-case-exclude="*[graphical]*"` |
  | **Working directory**     | `$ProjectFileDir$\install\windows-Release\bin`    |
  | **Environment variables** | `PATH=..\lib\;$Path$`                             |
  | **Before launch**         | `install` + `Prepare test run`                    |
 
- Test Debug

  | **Type**                  | Doctest                                           |
  |---------------------------|---------------------------------------------------|
  | **Target**                | `cosmoscout`                                      |
  | **Program arguments**     | `--run-tests --test-case-exclude="*[graphical]*"` |
  | **Working directory**     | `$ProjectFileDir$\install\windows-Debug\bin`      |
  | **Environment variables** | `PATH=..\lib\;$Path$`                             |
  | **Before launch**         | `install` + `Prepare test run`                    |

#### Linux
- Run Release

  | **Type**                  | CMake Application                                                                                                    |
  |---------------------------|----------------------------------------------------------------------------------------------------------------------|
  | **Target**                | `cosmoscout`                                                                                                         |
  | **Executable**            | `$ProjectFileDir$/install/linux-Release/bin/cosmoscout`                                                              |
  | **Program arguments**     | `--settings=../share/config/simple_desktop.json -vista vista.ini`                                                    |
  | **Environment variables** | `LD_LIBRARY_PATH=../lib:../lib/DriverPlugins:$LD_LIBRARY_PATH;VISTACORELIBS_DRIVER_PLUGIN_DIRS=../lib/DriverPlugins` |
  | **Before launch**         | `install`                                                                                                            |
  
- Run Debug

  | **Type**                  | CMake Application                                                                                                    |
  |---------------------------|----------------------------------------------------------------------------------------------------------------------|
  | **Target**                | `cosmoscout`                                                                                                         |
  | **Executable**            | `$ProjectFileDir$/install/linux-Debug/bin/cosmoscout`                                                                |
  | **Program arguments**     | `--settings=../share/config/simple_desktop.json -vista vista.ini`                                                    |
  | **Environment variables** | `LD_LIBRARY_PATH=../lib:../lib/DriverPlugins:$LD_LIBRARY_PATH;VISTACORELIBS_DRIVER_PLUGIN_DIRS=../lib/DriverPlugins` |
  | **Before launch**         | `install`                                                                                                            |

- Test Release

  | **Type**                  | Doctest                                                        |
  |---------------------------|----------------------------------------------------------------|
  | **Target**                | `cosmoscout`                                                   |
  | **Program arguments**     | `--run-tests --test-case-exclude="*[graphical]*"`              |
  | **Working directory**     | `$ProjectFileDir$/install/linux-Release/bin`                   |
  | **Environment variables** | `LD_LIBRARY_PATH=../lib:../lib/DriverPlugins:$LD_LIBRARY_PATH` |
  | **Before launch**         | `install` + `Prepare test run`                                 |

- Test Debug

  | **Type**                  | Doctest                                                        |
  |---------------------------|----------------------------------------------------------------|
  | **Target**                | `cosmoscout`                                                   |
  | **Program arguments**     | `--run-tests --test-case-exclude="*[graphical]*"`              |
  | **Working directory**     | `$ProjectFileDir$/install/linux-Debug/bin`                     |
  | **Environment variables** | `LD_LIBRARY_PATH=../lib:../lib/DriverPlugins:$LD_LIBRARY_PATH` |
  | **Before launch**         | `install` + `Prepare test run`                                 |

### Plugins
For CLion, we can recommend these plugins for the development of CosmoScout VR:
- [.gitignore](https://plugins.jetbrains.com/plugin/7495--ignore/)
- [CodeGlance Pro](https://plugins.jetbrains.com/plugin/18824-codeglance-pro)
- [GLSL](https://plugins.jetbrains.com/plugin/18470-glsl)
- [Rainbow Brackets](https://plugins.jetbrains.com/plugin/10080-rainbow-brackets)


## <img src="https://simpleicons.org/icons/cplusplus.svg" alt="Simple Icons" width=24 height=18> Visual Studio (Windows only)

First, you should follow the [Generic Build Instructions](install.md) for Windows. Once CosmoScout VR has been compiled successfully, you can simply open the `cosmoscout-vr.sln` solution in either `build\windows-Debug` or `build\windows-Release`.

Due to the build process of CosmoScout VR, this solution can only be used for Debug or Release mode respectively. Therefore you have to select the corresponding configuration type in Visual Studio.

## <img src="https://simpleicons.org/icons/vscodium.svg" alt="Simple Icons" width=24 height=18> Visual Studio Code (Linux & Windows)

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
      "command": "./make.sh -DCOSMOSCOUT_UNIT_TESTS=On",
      "options": {
        "cwd": "${workspaceFolder}"
      },
      "problemMatcher": [
        "$gcc"
      ],
      "windows": {
        "command": ".\\make.bat -DCOSMOSCOUT_UNIT_TESTS=On",
        "options": {
          "env": {
            "BOOST_ROOT": "C:\\local\\boost_1_69_0"
          }
        }
      }
    },
    {
      "label": "Make (Debug)",
      "type": "shell",
      "command": "./make.sh -DCOSMOSCOUT_UNIT_TESTS=On",
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
        "command": ".\\make.bat -DCOSMOSCOUT_UNIT_TESTS=On",
        "options": {
          "env": {
            "BOOST_ROOT": "C:\\local\\boost_1_69_0"
          }
        }
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
      "label": "Run CosmoScout VR (Release)",
      "type": "shell",
      "command": "install/linux-Release/bin/start.sh",
      "options": {
        "cwd": "${workspaceFolder}"
      },
      "problemMatcher": [],
      "windows": {
        "command": ".\\install/windows-Release/bin/start.bat"
      }
    },
    {
      "label": "Run CosmoScout VR (Debug)",
      "type": "shell",
      "command": "install/linux-Debug/bin/start.sh",
      "options": {
        "cwd": "${workspaceFolder}"
      },
      "problemMatcher": [],
      "windows": {
        "command": ".\\install/windows-Debug/bin/start.bat"
      }
    },
    {
      "label": "Run Tests (Release)",
      "type": "shell",
      "command": "install/linux-Release/bin/run_tests.sh",
      "options": {
        "cwd": "${workspaceFolder}"
      },
      "problemMatcher": [],
      "windows": {
        "command": ".\\install/windows-Release/bin/run_tests.bat"
      }
    },
    {
      "label": "Run Tests (Debug)",
      "type": "shell",
      "command": "install/linux-Debug/bin/run_tests.sh",
      "options": {
        "cwd": "${workspaceFolder}"
      },
      "problemMatcher": [],
      "windows": {
        "command": ".\\install/windows-Debug/bin/run_tests.bat"
      }
    }
  ]
}
```

If you are on Windows, you may have to replace the `"BOOST_ROOT"` environment variable in this file.

With this file in place, you can press `Ctrl+Shift+P` and select `Tasks: Run Task`. Now you can first select `Make Externals (Release)`, then `Make (Release)` and later `Run CosmoScout VR`.

> ![TIP]
> 
> **(Linux only):** You can use [ccache](https://ccache.dev/) to considerably speed up build times. You just need to replace the commands with `./make_externals.sh -G "Unix Makefiles" -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache` and `./make.sh -G "Unix Makefiles" -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache` respectively._

> ![TIP]
>
> **(Windows only):** You can use [clcache](https://github.com/frerich/clcache) to considerably speed up build times. You just need to call `make_externals.bat -G "Visual Studio 15 Win64" -DCMAKE_VS_GLOBALS=CLToolExe="clcache.exe;TrackFileAccess=false"` and `make.bat -G "Visual Studio 15 Win64" -DCMAKE_VS_GLOBALS="CLToolExe=clcache.exe;TrackFileAccess=false"` respectively._

### `.vscode/c_cpp_properties.json`

```json
{
    "configurations": [
        {
            "name": "Linux",
            "compileCommands": "${workspaceRoot}/build/linux-Release/compile_commands.json",
            "browse": {
                "databaseFilename": "${workspaceRoot}/.vscode/browse-linux.VC.db"
            },
            "includePath": [
                "${workspaceRoot}/build/linux-Release/**"
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
                "${workspaceRoot}/build/windows-Release/**",
                "${workspaceRoot}/install/windows-externals-Release/include"
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
      "program": "${workspaceFolder}/install/linux-Debug/bin/cosmoscout",
      "args": [
        "--settings=../share/config/simple_desktop.json",
        "-vistaini vista.ini"
      ],
      "stopAtEntry": false,
      "cwd": "${workspaceFolder}/install/linux-Debug/bin",
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
      "program": "${workspaceFolder}/install/windows-Debug/bin/cosmoscout",
      "args": [
        "--settings=../share/config/simple_desktop.json",
        "-vistaini vista.ini"
      ],
      "stopAtEntry": false,
      "cwd": "${workspaceFolder}/install/windows-Debug/bin",
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

<p align="center"><img src ="img/hr.svg"/></p>
<p align="center">
  <a href="install.md">&lsaquo; Generic Build Instructions</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="README.md">&#8962; Help Index</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="using.md">Using CosmoScout VR &rsaquo;</a>
</p>
