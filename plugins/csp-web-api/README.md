# Web API for CosmoScout VR

A CosmoScout VR plugin which provides an HTTP protocol for controlling the application. This plugin is built as part of CosmoScout's build process. See the [main repository](https://github.com/cosmoscout/cosmoscout-vr) for instructions.

This is a default plugin of CosmoScout VR. Hence, any **issues should be reported to the [main issue tracker](https://github.com/cosmoscout/cosmoscout-vr/issues)**. There you can add a label indicating which plugins are affected.

### ⚠️ **WARNING** ⚠️
This plugin deliberately exposes the Javascript API of CosmoScout VR over the Rest API. **This can be abused and poses a security risk for the server and the host!** The plugin should only be used in internal networks, where the users are trusted.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`.

```javascript
{
  ...
  "plugins": {
    ...
    "csp-web-api": {
      "port": 9001,
      "page": "../share/resources/gui/example-web-frontend.html"
    }
  }
}
```

**More in-depth information and some tutorials will be provided soon.**
