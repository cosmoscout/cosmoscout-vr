# Frame-Timings for CosmoScout VR

A plugin which uses the built-in timer queries of CosmoScout VR to draw on-screen live frame timing statistics.
This plugin can also be used to export recorded time series to a set of CSV files (in microseconds).

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
* When the timer queries are enabled, you can show the on-screen statistics. Move the pointer over the statistics window to see more details.
* You can also start a recording by clicking the big Record-Frame-Timings-button. Once you finish the recording, several CSV files will be written to a directory called `csp-timings/<current date>` in CosmoScout VR's `bin` directory. The files prefixed with `gpu-` contain GPU timing information, the others contain CPU timing data. The timing data is sorted by nesting level of the timed ranges - this means that the data in one file can be safely accumulated for one frame as it does not contain overlapping ranges. If timing ranges with the same name have been measured in one frame, their data will be accumulated in the files. 