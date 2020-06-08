# Anchor Labels for CosmoScout VR

A CosmoScout VR plugin which draws a clickable label at each celestial anchor. When activated, the user automatically travels to the according body. The size and overlapping-behavior of the labels can be adjusted.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`.
The given values present some good starting values for your customization:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-anchor-labels": {
      "enabled": true,               // If true the labels will be displayed at startup.
      "enableDepthOverlap": true,    // If true the labels will ignore depth for collision.
      "ignoreOverlapThreshold": 0.1, // How close labels can get without one being disabled.
      "labelScale": 1.2,             // The size of the labels.
      "depthScale": 1.0,             // Determines how much smaller far away labels are.
      "labelOffset": 0.2             // How far over the anchor's center the label is placed.
     }
  }
}
```

**More in-depth information and some tutorials will be provided soon.**
