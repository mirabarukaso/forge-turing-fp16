# Frontend flow: release Forge weights between requests

Base URL is the WebUI origin, for example `http://127.0.0.1:50000`. The launch arguments must include `--api`.

The extension cannot make a float32 reload use the float16 weight budget. `POST /sdapi/v1/unload-checkpoint` removes weights that are still resident from the previous request. The next generation can still offload part of a large UNet when compute is float32.

## Endpoints

`POST /sdapi/v1/unload-checkpoint` with an empty body. It calls `unload_model_weights()`:

- unloads every model Forge is tracking, including the UNet, text encoder, and VAE
- drops the checkpoint object and clears `forge_hash`
- runs CUDA `empty_cache` and garbage collection

The response body is `{}`.

`GET /sdapi/v1/memory` returns `ram` and `cuda`. Compare `cuda.system.free` before and after the unload. Values are bytes. `cuda.allocated.current` should fall. `cuda.reserved.current` can stay high until a later allocation. That is not a leftover UNet.

`override_settings_restore_afterwards: true` restores option values after sampling. It does not unload. A later log line that selects the UI checkpoint is that restore, not a second sample.

## Sequence

1. Poll `GET /sdapi/v1/progress` until the job is idle. Do not unload during sampling.
2. `GET /sdapi/v1/memory` and store `cuda.system.free`.
3. When the next job turns `forge_compute_fp16` or `forge_sage_fp16_cast` off, or changes `sd_model_checkpoint` or `forge_additional_modules`:
   - `POST /sdapi/v1/unload-checkpoint`
   - `GET /sdapi/v1/memory` again. `cuda.system.free` should increase. If it does not, wait briefly and call unload once more. Do not loop.
4. `POST /sdapi/v1/txt2img` or `POST /sdapi/v1/img2img` with `override_settings` and `override_settings_restore_afterwards: true`.

Turning both switches off inside that same `override_settings` already makes the extension reload during the request. Step 3 clears the previous request's text encoder and VAE before that reload. It does not stop float32 partial offload on the new load.

Skip step 3 when the checkpoint, the extra modules, and both switches stay the same. Unload then only forces a full reload.

## Setting keys

These `override_settings` keys exist only while the extension is loaded:

- `forge_compute_fp16` boolean
- `forge_sage_fp16_cast` boolean

Checkpoint keys are Forge's own:

- `sd_model_checkpoint` string
- `forge_additional_modules` list

`POST /sdapi/v1/options` writes saved options and applies on the next generation. It does not unload. Use `override_settings` on the generate call when the UI should revert afterward.
