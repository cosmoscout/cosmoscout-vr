# Custom Web UI for CosmoScout VR

A CosmoScout VR plugin which allows adding custom HTML-based UI elements as sidebar-tabs, as floating windows or into free space. This plugin is built as part of CosmoScout's build process. See the [main repository](https://github.com/cosmoscout/cosmoscout-vr) for instructions.

This is a default plugin of CosmoScout VR. Hence, any **issues should be reported to the [main issue tracker](https://github.com/cosmoscout/cosmoscout-vr/issues)**. There you can add a label indicating which plugins are affected.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`.
The given values are just some examples, feel free to add your own items:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-custom-web-ui": {
      "sidebar-items": [
        {
          "name": "Spotify",
          "icon": "queue_music",
          "html": "<iframe src='https://open.spotify.com/embed/playlist/2xl3sX0pZajy1XOogLpc5m' width='100%' height='380' frameborder='0' allowtransparency='true' allow='encrypted-media'></iframe>"
        }
      ],
      "space-items": [
        {
          "center": "Earth",
          "icon": "IAU_Earth",
          "longitude": 14,
          "latitude": 51,
          "elevation": 100,
          "scale": 1000,
          "width": 800,
          "height": 600,
          "html": "<iframe width='100%' height='100%' src='https://www.dlr.de' frameborder='0'></iframe>"
        }
      ],
      "window-items": [
        {
          "name": "Wikipedia",
          "icon": "language",
          "html": "<iframe style='height: 100%; width: 100%; min-height: 200px; min-width: 300px; border: none' src='https://www.wikipedia.org'></iframe>"
        }
      ]
    },
    ...
  }
}
```

**More in-depth information and some tutorials will be provided soon.**

## MIT License

Copyright (c) 2020 German Aerospace Center (DLR)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
