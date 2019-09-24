<p align="center"> 
  <img src ="img/banner-sunset.jpg" />
</p>

# Releases of CosmoScout VR

Releases are [published on Github](https://github.com/cosmoscout/cosmoscout-vr/releases).
The progress of future releases is tracked with [Github Milestones](https://github.com/cosmoscout/cosmoscout-vr/milestones).
Submitted [issues](https://github.com/cosmoscout/cosmoscout-vr/issues) will be assigned to a specific release (depending on their importance and complexity).

[Github Projects](https://github.com/cosmoscout/cosmoscout-vr/projects) will be used for high-level planning of complex features, such as network synchronization or photorealistic HDR rendering.

## Version Numbers

Releases in the 1.x.x series will most likely have a lot of breaking API changes, as the software becomes more and more mature.
However, starting from version 2.0.0, version numbers of CosmoScout VR will be assigned according to the [Semantic Versioning](https://semver.org/) scheme.
This means, given a version number MAJOR.MINOR.PATCH, we will increment the:

1. MAJOR version when we make incompatible API changes,
2. MINOR version when we add functionality in a backwards compatible manner, and
3. PATCH version when we make backwards compatible bug fixes.

## Creating Releases

When a new version of CosmoScout VR is released, the following steps are performed.

```bash
git checkout develop
git submodule update --init
```

First, the [changelog.md](https://github.com/cosmoscout/cosmoscout-vr/blob/develop/docs/changelog.md) has to be updated.
Based on the commits since the last release and the completed project board, a list of changes is compiled.
When this is done, the file has to be comitted:

```bash
git add docs/changelog.md
git commit -m ":memo: Update changelog.md"
```

Then edit the [project(... VERSION ...)](https://github.com/cosmoscout/cosmoscout-vr/blob/develop/CMakeLists.txt#L8) in the main `CMakeLists.txt` file according to the new version number.
Afterwards, the change has to be comitted:

```bash
git add CMakeLists.txt
git commit -m ":tada: Bump version number"
```

Then we create a new git tag and merge this state to the `master` branch.

```bash
git push origin develop
git tag v<new version number>
git checkout master
git merge develop
git push origin v<new version number>
git push origin master
```

The default downloads for tags on Github do not contain git submodules.
Therefore we create a seperate archive which contains all the submodule code. Here is the command to create the archive:

```bash
externals/git-archive-all.sh/git-archive-all.sh --prefix cosmoscout-vr/ source-with-submodules.tar.gz
```

The resulting file `source-with-submodules.tar.gz` is then uploaded to the new release on Github.

<p align="center">
  <a href="citation.md">&lsaquo; How to cite CosmoScout VR</a>
  <img src ="img/nav-vspace.svg"/>
  <a href="README.md">&#8962; Help Index</a>
</p>

<p align="center"><img src ="img/hr.svg"/></p>
