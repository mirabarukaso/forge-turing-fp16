# Forge FP16 and unloading VRAM

Start Forge Neo with `--api`. The two FP16 switches come from this extension. If the extension is not loaded, `forge_compute_fp16` and `forge_sage_fp16_cast` do not exist.

The client in this note sends only `POST /sdapi/v1/txt2img`. A normal generation and a regional generation use the same overrides and the same unload decision. The base URL is the WebUI origin, for example `http://127.0.0.1:7860`.

## Three controls

WebUI stores the weight dtype separately from the two compute switches.

`forge_unet_storage_dtype` must be one of these six strings:

- `Automatic`
- `Automatic (fp16 LoRA)`
- `float8-e4m3fn`
- `float8-e4m3fn (fp16 LoRA)`
- `float8-e5m2`
- `float8-e5m2 (fp16 LoRA)`

The two checkboxes:

| UI | Request field | Effect |
|---|---|---|
| PyTorch_FP16 | `forge_compute_fp16` | Compute on the loaded UNet switches to fp16. Weight storage stays as it is. |
| Sage_FP16 | `forge_sage_fp16_cast` | Cast q/k/v to fp16 before SageAttention, then cast the output back. |

Neither checkbox forces the other. With only Sage on, compute dtype stays unchanged and only the attention inputs are cast. While either checkbox is on, the storage dtype sent with the request is `Automatic`. Changing storage dtype in the UI while a checkbox is still on also goes back to `Automatic`.

Checking either box prompts the user to install this extension, once per process. Sage also needs SageAttention 1.0.6 or 2.2.

An A1111 backend does not send these three keys. If a checkbox is still on, the log is `FP16 switches skipped: backend is not Forge`.

## Order of one generation

1. `GET /sdapi/v1/options`. The client treats the backend as Forge when `forge_additional_modules` is present.
2. When the model, VAE, additional modules, or the filename rule changed, `POST /sdapi/v1/options` writes those loading options. This call does not unload VRAM, and it does not write the two FP16 switches.
3. Decide whether to unload, using the next section. When an unload is required, follow "Unload sequence".
4. `POST /sdapi/v1/txt2img`.

`POST /sdapi/v1/options` only updates Forge's own loading options. The FP16 switches go in `override_settings` on the generation request, so the saved options can be restored after sampling and that restore does not convert the weights already loaded.

## Generation request

`override_settings_restore_afterwards` is always `true`. The three dtype keys are sent every time. The checkpoint and the additional modules are added to the same `override_settings` only when they differ from the current options.

```json
{
  "prompt": "a cat",
  "override_settings_restore_afterwards": true,
  "override_settings": {
    "sd_model_checkpoint": "model.safetensors",
    "forge_additional_modules": ["clip.safetensors", "vae.safetensors"],
    "forge_compute_fp16": false,
    "forge_sage_fp16_cast": true,
    "forge_unet_storage_dtype": "Automatic"
  }
}
```

When both switches are off and storage dtype is FP8, those fields are:

```json
{
  "forge_compute_fp16": false,
  "forge_sage_fp16_cast": false,
  "forge_unet_storage_dtype": "float8-e4m3fn"
}
```

`override_settings_restore_afterwards: true` writes the options back after sampling. It does not unload the UNet, the text encoder, or the VAE. A later log line that selects the UI checkpoint is that restore, not a second sample.

All four combinations are valid:

| forge_compute_fp16 | forge_sage_fp16_cast | Result |
|---|---|---|
| true | true | The whole network computes in fp16. Attention inputs are already fp16, so the Sage cast does not change the values. |
| true | false | The whole network computes in fp16, with no Sage cast. |
| false | true | Compute dtype stays as it is. Only q/k/v are cast to fp16 and back. |
| false | false | Neither switch applies. |

## When to unload

The process remembers the four values actually sent last time: `sd_model_checkpoint`, `forge_additional_modules`, `forge_compute_fp16`, and `forge_sage_fp16_cast`. Compare those with the values about to be sent, not with the current `GET /sdapi/v1/options` body. After a restore, the saved switches are off again while the weights are still the ones loaded by the previous override. Comparing against the saved options unloads on every request even when the model did not change.

| This request compared with the last one | Action |
|---|---|
| Both switches, the checkpoint, and the additional modules are the same | Skip. Unloading here only forces a full reload. |
| Only `forge_unet_storage_dtype` changed | Skip |
| Either switch goes from off to on | Skip |
| `forge_compute_fp16` goes from on to off | Unload first |
| `forge_sage_fp16_cast` goes from on to off | Unload first |
| The checkpoint changed | Unload first |
| The additional-module list changed | Unload first |

One request can match more than one row. On the first generation there is no previous record. Unload only when this checkpoint or these additional modules differ from the options that were current before `setModel`. A switch going from off to on does not unload.

The unload drops the text encoder and VAE left by the previous request, and any UNet still resident. When both switches are turned off in the same `override_settings`, the extension reloads during that request. The unload happens before that reload. After compute returns to float32, a large UNet can still load only in part. Unloading does not make that float32 load use the fp16 weight budget.

After the client process restarts, the previous record is gone. The first generation after a restart can unload only from a model difference against the current options. Turning a switch off is recognized only after a generation in this same process already sent it as on.

## Unload sequence

Do not unload while sampling. Read progress first:

`GET /sdapi/v1/progress`

Continue only when `state.job_count` is `0`. Poll every 500ms, for at most 30 seconds. On timeout, skip the unload, log `progress still busy, skipped`, and generate anyway.

Once the job is idle:

1. `GET /sdapi/v1/memory` and store `cuda.system.free`. The unit is bytes.
2. `POST /sdapi/v1/unload-checkpoint` with body `{}`. A successful response body is `{}`. Time out after 60 seconds.
3. `GET /sdapi/v1/memory` again. `cuda.system.free` should be larger.
4. If it did not grow, wait 500ms, call `POST /sdapi/v1/unload-checkpoint` once more, and read memory once more. Stop there. Do not loop.
5. `POST /sdapi/v1/txt2img`.

`POST /sdapi/v1/unload-checkpoint` calls `unload_model_weights()`:

- unloads every model Forge is tracking, including the UNet, the text encoder, and the VAE
- drops the checkpoint object and clears `forge_hash`
- runs CUDA `empty_cache` and garbage collection

`cuda.allocated.current` should fall. `cuda.reserved.current` can stay high until a later allocation. That is not a leftover UNet.

If the unload call or the memory call fails, log it and continue with generation.

## Logs

No log line when nothing changed, when only the storage dtype changed, or when a switch goes from off to on.

When an unload runs, write one `[WebUI]` line with every reason, the before and after values, and `cuda.system.free` before and after:

```text
[WebUI] unload checkpoint: forge_compute_fp16 true -> false; checkpoint model-a.safetensors -> model-b.safetensors; forge_additional_modules clip.safetensors, vae.safetensors -> vae.safetensors cuda.system.free 1073741824 -> 6442450944
```

A second unload is its own line:

```text
[WebUI] unload checkpoint again: cuda.system.free 1073741824 -> 1073741824
```

When free VRAM cannot be read, write `unavailable` in that position.
