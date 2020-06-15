# Recorder for CosmoScout VR

A CosmoScout VR plugin which allows basic capturing of high-quality videos.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`.
The given values present the default values and are all optional:

```javascript
{
  ...
  "plugins": {
    ...
    "csp-recorder": {
      "webAPIPort":     9001,    // Port of csp-web-api.
      "recordObserver": true,   // If true, the observer transformation will be recorded for each frame.
      "recordTime":     true,   // If true, the simulation time will be recorded for each frame.
      "recordExposure": false  // If true, the exposure of each frame will be recorded. Requires HDR mode.
     },
     "csp-web-api": {           // This plugin is required by csp-recorder.
      "port": 9001
     }
  }
}
```

## Usage

The capturing is done in three steps:
1. **Record Something:** Setup a scene with as little plugins as possible as frame rates matter.
Maybe it's a good idea to only render simple planets.
The hit the record button beneath the timeline and fly around.
When finished, hit the record button once more.
2. **Capture the Frames:** Step 1 has produced a python script called `recording-<current date>.py` next to the cosmoscout executable.
This can be executed to capture the frames!
Configure now your scene to look as good as possible - enable all the fancy plugins!
Move all quality sliders to their upper limit!
And make sure to have the `csp-web-api` plugin enabled, as this is required for the capturing.
Then run something like the following in CosmoScout's `bin` directory:
   ```bash
   # This is only required once
   python3 -m pip install requests
 
   mkdir recording && cd recording
   python3 ../recording<current date>.py
   ```
3. **Encode the Frames:** Using something like `ffmpeg`, the individual frames can be merged to a video file.
Here is an example:
   ```bash
   ffmpeg -f image2 -framerate 60 -i frame_%d.png -c:v libx264 -preset veryslow  -qp 8 -pix_fmt yuv420p recording.mp4
   ```
