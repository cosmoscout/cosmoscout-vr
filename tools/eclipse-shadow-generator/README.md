# Eclipse Shadow Map Generator

This tool can be used to generate the eclipse shadow maps used by CosmoScout VR.
To learn about the usage, you can issue this command:


```bash
./eclipse-shadow-generator --help
```

Here are some other examples to get you started:

```bash
# This simple command creates the default eclipse shadow map of CosmoScout VR
./eclipse-shadow-generator

# Here are some other examples
./eclipse-shadow-generator --mode circles --output "circles.hdr"
./eclipse-shadow-generator --mode smoothstep --output "smoothstep.hdr"
./eclipse-shadow-generator --mode linear --with-umbra --mapping-exponent 5 --output "linear_with_umbra.hdr"
```