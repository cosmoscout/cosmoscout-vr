<!-- 
SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
SPDX-License-Identifier: CC-BY-4.0
 -->
 
 # Atmosphere Lookup-Texture Generator

This tool can be used to generate the precomputed atmosphere data required for the advanced atmosphere model used by CosmoScout VR.

## Building

**Per default, the atmosphere generator is not built.
To build it, you need to pass `-DCS_ATMOSPHERE_GENERATOR=On` in the make script.**

## Usage

Once compiled, you'll need to set the library search path to contain the `install/<os>-<build_type>/lib` directory.
This depends on where the `atmosphere-generator` is installed to, but this may be something like this:

```powershell
# For powershell
$env:Path += ";..\lib"

# For bash
export LD_LIBRARY_PATH=../lib:$LD_LIBRARY_PATH
```

To learn about the usage of `atmosphere-generator`, you can now issue this command:


```bash
./atmosphere-generator --help
```

Here are some other examples to get you started:

```bash

```