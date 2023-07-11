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
      "enabled": <bool>,           // Toggle whether the scenario is visible/active
      "debug": <bool>,             // Toggle debug mode, where all stages are visible
      "otherScenarios": [          // List of other scenario configs related to the current scenario
        {
          "name": <string>,        // Name of the scenario
          "path": <string>         // Path to the config (e.g. "../share/scenes/scenario_name.json")
        },
        ...
      ],
      "stages": [                  // List of stages in each scenario
        {
          "type": <string>,        // Type of a stage (enum) see below for stage types
          "bookmark": <string>,    // Name of the bookmark used for stage position
          "scale": <float>         // Scaling factor for the size of the web view
        },
        ...
      ]
     }
  }
}
```

### Stage Types

| Type             | Description |
|:-----------------|:------------|
| `checkpoint`     | Draws a gate the user has to move through in the correct direction. |
| `requestFMS`     | Draws a panel to requests the user to submit a score on the FMS. |
| `switchScenario` | Draws a panel displaying the list of `otherScenarios` allowing the user to switch to a different scenario. |
