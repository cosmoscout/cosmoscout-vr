# Contributing

We are happy to receive contributions to CosmoScout VR in the form of **merge requests** via Github. Feel free to fork the repository, implement your changes and create a merge request to the `develop` branch.

**Bugs** should be reported via Github issues.

The `master` branch should always contain stable code. New features and bug fixes are implemented in `feature/*` branches and are merged to `develop` once they are finished. When a new milestone is reached, the content of `develop` will be merged to `master`.

## Some Coding Guidelines

* Each header file should contain an include guard. For CosmoScout VR classes the naming scheme should be `VP_{NAMESPACE}_{FILNAME}_HPP` and for plugins it should be `VPP_{PLUGIN}_{FILNAME}_HPP`.
* Class names should be written in CamelCase (e.g. `MyClass`).
* Class methods should be written in small camelCase (e.g. `doSomething()`). 
* Class members should start with a small `m` and continue in CamelCase (e.g. `mMyClassMember`). 
* Apply clang-format before you create a merge request (either setup your IDE to do this or use the `clang-format.sh` script)
* Never use `using namespace`.
* Use features of modern C++11 / C++14 / C++17 (e.g. range-based for-loops, std::optional, std::variant, ...)!

## DLR Individual Contributor License Agreement

Before we can accept your merge request, you have to print, sign, scan and send the [DLR Individual Contributor License Agreement](CLA.md) via e-mail to cosmoscout@dlr.de.

## Git Commit Messages

Commits should start with a Capital letter and should be written in present tense (e.g. __:tada: Add cool new feature__ instead of __:tada: Added cool new feature__).
It's a great idea to start the commit message with an applicable emoji. This does not only look great but also makes you rethink what to add to a commit.
* :tada: `:tada:` when when adding a cool new feature
* :wrench: `:wrench:` when refactoring / improving a small piece of code
* :hammer: `:hammer:` when refactoring / improving large parts of the code
* :sparkles: `:sparkles:` when applying clang-format
* :art: `:art:` improving / adding assets like textures or 3D-models
* :rocket: `:rocket:` when improving performance
* :memo: `:memo:` when writing docs
* :beetle: `:beetle:` when fixing a bug
* :green_heart: `:green_heart:` when fixing the CI build
* :arrow_up_small: `:arrow_up_small:` when adding / upgrading dependencies
* :arrow_down_small: `:arrow_down_small:` when removing / downgrading dependencies
* :fire: `:fire:` when removing files
* :truck: `:truck:` when moving / renaming files or namespaces

A good way to enforce this on your side is to use a `commit-hook`. To do this, paste the following script into `.git/hooks/commit-msg`.

``` bash
#!/bin/bash

# regex to validate in commit msg
commit_regex='(:(tada|wrench|hammer|sparkles|art|rocket|memo|beetle|green_heart|arrow_up_small|arrow_down_small|fire|truck):(.+))'
error_msg="Aborting commit. Your commit message is missing an emoji as described in CONTRIBUTING.md"

if ! grep -xqE "$commit_regex" "$1"; then
    echo "$error_msg" >&2
    exit 1
fi
```

And make sure that it is executable:

``` bash
chmod +x .git/hooks/commit-msg
```
