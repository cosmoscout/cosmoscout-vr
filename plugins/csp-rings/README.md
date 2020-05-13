# Rings for CosmoScout VR

A CosmoScout VR plugin which can draw simple rings around celestial bodies. The rings can be configured with a inner and a outer radius and a texture. This plugin is built as part of CosmoScout's build process. See the [main repository](https://github.com/cosmoscout/cosmoscout-vr) for instructions.

This is a default plugin of CosmoScout VR. Hence, any **issues should be reported to the [main issue tracker](https://github.com/cosmoscout/cosmoscout-vr/issues)**. There you can add a label indicating which plugins are affected.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-rings": {
      "rings": {
        <anchor name>: {
          "texture": <path to the texture>,   // The texture should be a cross section
                                              // of the ring.
          "innerRadius": <meters from planet center>,
          "outerRadius": <meters from planet center>
        },
        ... <more rings> ...
      }
    }
  }
}
```

**More in-depth information and some tutorials will be provided soon.**
