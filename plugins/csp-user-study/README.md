# User Studies for CosmoScout VR

A CosmoScout VR plugin which allows for Pre-programmed scenarios, to study 6-DoF-Movement in VR.

## Configuration

This plugin can be enabled with the following configuration in your `settings.json`.
The given values present a basic scenario:

```json
{
  ...
  "plugins": {
    ...
    "csp-user-study": {
      "enabled": true,
      "otherScenarios": {                       // List of other scenarios' configs
        "scenario_A": "../share/scenes/basicScenario1.json",
        "scenario_B": "../share/scenes/basicScenario2.json"
      },
      "stages": [                               // List of stages in each scenario
        {
          "type": "checkpoint",                 // Checkpoint type stage, subject must pass through checkpoint to pass
          "bookmark": "name",                   // Name of the bookmark used for positional data
          "scale": 42.0                         // Scaling factor for the gate size
        },
        {
          "type": "requestFMS"                  // FMS report type stage, subject must select a score on the FMS to pass
        },
        {
          "type": "requestCOG"                  // CoG report type stage, subject must record CoG to pass
        },
        {
          "type": "switchScenario"              // Stage type to switch to another scenario, subject must select new scenario from "otherScenarios" list
        }
      ]
     }
  }
}
```
