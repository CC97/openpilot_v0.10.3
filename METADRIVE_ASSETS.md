# MetaDrive Assets

This openpilot checkout uses MetaDrive from the local virtual environment:

```bash
.venv/lib/python3.12/site-packages/metadrive
```

The comma.ai minimal MetaDrive assets package is small and does not include the full vehicle models. If `render_vehicle=True` fails with an error like:

```text
OSError: Could not load model file(s): .../assets/models/ferra/right_tire_front.gltf
```

pull the full upstream MetaDrive assets instead.

## Download Full Assets

First make sure the local MetaDrive asset puller points to the full asset package:

```bash
sed -i 's#https://github.com/commaai/metadrive/releases/download/MetaDrive-minimal/assets.zip#https://github.com/metadriverse/metadrive/releases/download/MetaDrive-0.4.2.3/assets.zip#' .venv/lib/python3.12/site-packages/metadrive/pull_asset.py
```

Then download and overwrite the existing assets:

```bash
.venv/bin/python -m metadrive.pull_asset --update
```

The full assets package is much larger than the minimal one. After extraction, the assets directory should be around 168 MB.

## Verify

Check that the vehicle body and tire models exist:

```bash
find .venv/lib/python3.12/site-packages/metadrive/assets/models -maxdepth 2 -type f \
  \( -name 'vehicle.gltf' -o -name '*tire*.gltf' \) | sort
```

Expected files include:

```text
.venv/lib/python3.12/site-packages/metadrive/assets/models/ferra/vehicle.gltf
.venv/lib/python3.12/site-packages/metadrive/assets/models/ferra/right_tire_front.gltf
.venv/lib/python3.12/site-packages/metadrive/assets/models/ferra/right_tire_back.gltf
```

You can also check the directory size:

```bash
du -sh .venv/lib/python3.12/site-packages/metadrive/assets
```

## Notes

Do not upgrade `metadrive-simulator` casually with pip. This openpilot tree pins a specific MetaDrive wheel, and changing the package version may break the simulator bridge API.
