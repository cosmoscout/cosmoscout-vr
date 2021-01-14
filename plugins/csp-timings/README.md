# Frame-Timings for CosmoScout VR

A plugin which uses the built-in timer queries of CosmoScout VR to draw on-screen live frame timing statistics.
This plugin can also be used to export recorded time series to a CSV file (in milliseconds).

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`.
There are no configuration options, so this is all you need to do:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-timings": {},
    ...
  }
}
```

## Usage

Once the plugin is loaded, you can enable the timer queries in the sidebar tab "Frame Timing".
* When the timer queries are enabled, you can show the on-screen statistics.
* You can also start a recording by clicking the big Record-Frame-Timings-button. Once you finish the recording, a file called `timing-<current date>.csv` will be written to CosmoScout VR's `bin` directory.