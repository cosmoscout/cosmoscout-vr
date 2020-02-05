<p align="center"> 
  <img src ="img/banner-stars.jpg" />
</p>

# Contributing to CosmoScout VR

Whenever you encounter a :beetle: **bug** or have :tada: **feature request**, 
report this via [Github issues](https://github.com/cosmoscout/cosmoscout-vr/issues).

We are happy to receive contributions to CosmoScout VR in the form of **pull requests** via Github.
Feel free to fork the repository, implement your changes and create a merge request to the `develop` branch.

Since CosmoScout VR uses plenty of git submodules, forking is not straight-forward.
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

:warning: _**Warning:** The guides below are only for forking the project on GitHub. If you plan to mirror the repository to some other hosting environment (e.g. GitLab), you will have to mirror all plugins since the plugin submodules are included with relative paths (see [.gitmodules](../.gitmodules))._

Depending on what you're planning to implement, you have several options here:

1. **Fix or implement something in the core engine:** In this case you have to fork only the [main repository of CosmoScout VR](https://github.com/cosmoscout/cosmoscout-vr). All plugin submodules will point to the repositories in the [CosmoScout organization](https://github.com/cosmoscout).
1. **Fix or implement something in an existing plugin:** In this case you have to fork only the corresponding plugin. You can still use the [main repository of CosmoScout VR](https://github.com/cosmoscout/cosmoscout-vr).
1. **Fix or implement something in an existing plugin which requires changes in the core engine:** In this case you have to do both, 1 and 2.
1. **Create a new plugin:** In this case you do not have to fork anything, except if your new plugin requires some changes to the core engine. Then you would go for option 1.

## 1. Forking only the main repository

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

## 2. Forking only a plugin

In this case, you have to clone CosmoScout's [main repository](https://github.com/cosmoscout/cosmoscout-vr) and fork the plugin you want to modify. Once all plugin submodules have been checked out, you can add your forked plugin as a second remote to the plugin submodule.

``` bash
git clone git@github.com:cosmoscout/cosmoscout-vr.git
cd cosmoscout-vr
git checkout develop
git submodule update --init
cd plugins/csp-whatever
git remote add myfork git@github.com:<your user name>/csp-whatever.git 
git checkout -b feature/your-new-feature

# ... do and commit your changes!

git push -u myfork feature/your-new-feature
```

When there were changes in the plugin's develop branch, you will need to merge those to your fork before creating a pull request:

``` bash
cd plugins/csp-whatever
git fetch origin
git merge origin/develop
```

Then you can create a pull request on GitHub to CosmoScout's develop branch.

## 3. Forking the main repository and a plugin

This basically combines both approaches above. First you have to fork the main repository and the plugin you want to modify. Then clone your forked repository and add a second remote to the plugin submodule pointing to your forked plugin repository.

``` bash
git clone git@github.com:<your user name>/cosmoscout-vr.git
cd cosmoscout-vr
git remote add upstream git@github.com:cosmoscout/cosmoscout-vr.git
git checkout develop
git submodule update --init
git checkout -b feature/your-new-feature

cd plugins/csp-whatever
git remote add myfork git@github.com:<your user name>/csp-whatever.git 
git checkout -b feature/your-new-feature

# ... do and commit your changes, in both, plugin and main repository
# in your plugin you can push with

git push -u myfork feature/your-new-feature

# in your main repository you can push with
git push origin feature/your-new-feature
```

Merging upstream develop changes to your main repository fork is done like this:

``` bash
git fetch upstream
git merge upstream/develop
```

Merging upstream develop changes to your plugin repository fork is done like this:

``` bash
cd plugins/csp-whatever
git fetch origin
git merge origin/develop
```

Once you are satisfied, you can create pull requests for both, your modified plugin and the main repository on GitHub.

## 4. Creating a new plugin

From a git-perspective, this is pretty straight-forward. Just create a git repository, name it `csp-<whatever>` and clone it to the `plugins/` directory of CosmoScout VR. For the beginning, you can copy the contents of another similar plugin to that directory. You will only need to add one line to the file `plugins/CMakeLists.txt` in order to include your new plugin to the built process.

<p align="center"><img src ="img/hr.svg"/></p>
<p align="center">
  <a href="configuring.md">&lsaquo; Configuring CosmoScout VR</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="README.md">&#8962; Help Index</a>
  <img src ="img/nav-vspace.svg"/>
</p>
