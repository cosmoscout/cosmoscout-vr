<p align="center"> 
  <img src ="img/banner-stars.jpg" />
</p>

# Contributing to CosmoScout VR

Whenever you encounter a :beetle: **bug** or have :tada: **feature request**, 
report this via [Github issues](https://github.com/cosmoscout/cosmoscout-vr/issues).

We are happy to receive contributions to CosmoScout VR in the form of **pull requests** via Github.
Feel free to fork the repository, implement your changes and create a merge request to the `develop` branch.
There is a [forking guide](#forking-cosmoscout-vr) available to get you started!

## Branching Guidelines

The development of CosmoScout VR follows a simplified version of **git-flow**: The `master` branch always contains stable code.
New features and bug fixes are implemented in `feature/*` branches and are merged to `develop` once they are finished.
When a new milestone is reached, the content of `develop` will be merged to `master` and a tag is created.

[Github Actions](https://github.com/cosmoscout/cosmoscout-vr/actions) are used for continuous integration.
All pull requests and pushes to `master` and `develop` are built automatically.
If you want to test a specific commit on any other branch, add **`[run-ci]`** to your commit message.

## Coding Guidelines

* Each header file should contain an include guard. For CosmoScout VR classes the naming scheme should be `CS_{NAMESPACE}_{FILNAME}_HPP` and for plugins it should be `CSP_{PLUGIN}_{FILNAME}_HPP`.
* Class names should be written in CamelCase (e.g. `MyClass`).
* Class methods should be written in small camelCase (e.g. `doSomething()`). 
* Class members should start with a small `m` and continue in CamelCase (e.g. `mMyClassMember`). 
* Apply clang-format before you create a merge request (either setup your IDE to do this or use the `clang-format.sh` script)
* Never use `using namespace`.
* Use features of modern C++11 / C++14 / C++17 (e.g. range-based for-loops, std::optional, std::variant, ...)!

### Git Commit Messages

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
* :heavy_check_mark: `:heavy_check_mark:` when working on tests
* :arrow_up_small: `:arrow_up_small:` when adding / upgrading dependencies
* :arrow_down_small: `:arrow_down_small:` when removing / downgrading dependencies
* :twisted_rightwards_arrows: `:twisted_rightwards_arrows:` when merging branches
* :fire: `:fire:` when removing files
* :truck: `:truck:` when moving / renaming files or namespaces

A good way to enforce this on your side is to use a `commit-hook`. To do this, paste the following script into `.git/hooks/commit-msg`.

``` bash
#!/bin/bash

# regex to validate in commit msg
commit_regex='(:(tada|wrench|hammer|sparkles|art|rocket|memo|beetle|green_heart|arrow_up_small|arrow_down_small|twisted_rightwards_arrows|fire|truck|heavy_check_mark):(.+))'
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

# Forking CosmoScout VR

This is pretty straight-forward. Just click the **Fork** button on the top right of this page. Then clone the forked repository, perform your changes, push to a feature branch and create a pull request to CosmoScout's develop branch.

``` bash
git clone git@github.com:<your user name>/cosmoscout-vr.git
cd cosmoscout-vr
git remote add upstream git@github.com:cosmoscout/cosmoscout-vr.git
git checkout develop
git submodule update --init
git checkout -b feature/your-new-feature

# ... do and commit your changes!

git push origin feature/your-new-feature
```

When there were changes in CosmoScout's develop branch, you will need to merge those to your fork before creating a pull request:

``` bash
git fetch upstream
git merge upstream/develop
```

Then you can create a pull request on GitHub to CosmoScout's develop branch.

## Creating a new plugin

From a git-perspective, this is pretty straight-forward. All you need to do is creating a new directory in `plugins/`. For the beginning, you can copy the contents of another similar plugin to that directory. It will be picked up automatically by the build system. The new directory will also be ignored by git (due to the toplevel `.gitignore` file). That means that you can use a git repository for your plugin.

<p align="center"><img src ="img/hr.svg"/></p>
<p align="center">
  <a href="configuring.md">&lsaquo; Configuring CosmoScout VR</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="README.md">&#8962; Help Index</a>
  <img src ="img/nav-vspace.svg"/>
</p>
