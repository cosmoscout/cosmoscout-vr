# Web API for CosmoScout VR

A CosmoScout VR plugin which provides an HTTP protocol for controlling the application.

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
