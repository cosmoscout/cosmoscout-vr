<!-- 
SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
SPDX-License-Identifier: CC-BY-4.0
 -->

# Demo Node Editor Plugin for CosmoScout VR

This plugin provides an example use case of the `csl-node-editor` plugin library.
For more information on this library, [have a look at it](../csl-node-editor/).


## Configuration & Usage

This plugin can be enabled with the following line in your `settings.json`:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-demo-node-editor": {
      "port": 9999
    }
  }
}
```

Once enabled, open a web browser and navigate to the web frontend at http://localhost:9999. You should see a grey grid canvas. Use your **right mouse button** to open the add-a-new-node menu. As long as CosmoScout VR is running, you can create an arbitrarily complex node layout. Closing and later re-opening the web frontend is possible; the graph layout will be restored automatically.