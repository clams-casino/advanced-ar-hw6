# Usage

Build it on the duckiebot. Assuming in the root of the repo run
```
dts devel build -f -H <duckiebot name>.local
```

To then run on the duckiebot, in the same directory, run
```
dts devel run -H <duckiebot name>.local
```

Publishes augmented images to `/<duckiebot name>/augmented_reality_apriltag/augemented_image/compressed`