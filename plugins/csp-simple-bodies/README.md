# Simple bodies for CosmoScout VR

A CosmoSout VR plugin which renders simple spherical celestial bodies. The bodies are drawn as an ellipsoid with an equirectangular texture.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-simple-bodies": {
      "bodies": {
        <anchor name>: {
          "texture": <path to surface texture>
        },
        ... <more bodies> ...
      }
    }
  }
}
```

**More in-depth information and some tutorials will be provided soon.**
