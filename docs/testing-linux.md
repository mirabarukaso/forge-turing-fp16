# Linux: tests

Run these by hand on Turing (sm75). The behavior checks match Windows. The launch command and package names do not.

Before starting, confirm `COMMANDLINE_ARGS` in `webui-user.sh` contains `--sage --skip-install` and does not contain `--reinstall-torch`. Start with `webui.sh`.

## Packages

```bash
venv/bin/python -m pip show sageattention triton
```

- `sageattention` is 2.2.0, and a restart does not replace it with a newer build.
- `triton` is older than 3.3. The package is not `triton-windows`.
- When the CUDA extension fails to load, the startup log contains `using Triton kernels`. That line is absent when the extension loads.

Run the Triton install from the manual page in bash. `"triton<3.3"` must be quoted so the shell does not try to read a file named `3.3`.

## Switches

The settings are on the Optimizations page:

- Diffusion compute in fp16
- Cast attention q/k/v to fp16 for SageAttention

Checks:

- Both off: the image matches a run without this extension.
- Compute fp16 only: the log contains `Diffusion compute dtype ... -> float16`. Two generations in a row do not reload the model on the second one. Turning the switch off reloads on the next generation.
- Sage cast only, and both switches on: sampling does not raise a Half/Float error.
- Generate once in low VRAM, then turn compute fp16 on, then generate again: no Half/Float error.
- After Hires fix starts its second pass, the compute dtype is still fp16.
- After the refiner swaps in another checkpoint, the compute dtype is still fp16.
- ControlNet sampling together with an fp16 UNet does not raise a dtype error.
- A Nunchaku model logs that fp16 compute was skipped, and generation does not crash.
- When the attention backend is not SageAttention, there is a warning, q/k/v are not cast, and generation still succeeds.
- An API request that sets the switch only through `override_settings` affects that request. After the request ends and the UI options are restored, the loaded model stays patched. The next generation reloads only when it sees the switch turned off.
- Disable this extension in the extension list and restart: the settings are gone, and behavior matches official Forge Neo.
