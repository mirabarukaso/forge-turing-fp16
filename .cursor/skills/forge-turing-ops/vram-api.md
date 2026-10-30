# Frontend VRAM sequence

The client procedure is [docs/api-vram.md](../../../docs/api-vram.md).

Compare the four values actually sent last time with the values about to be sent. Do not compare with `GET /sdapi/v1/options` after `override_settings_restore_afterwards` has turned the switches back off.

Unload before the next `txt2img` only when a switch goes from on to off, or when the checkpoint or `forge_additional_modules` changes. Skip when both switches go from off to on, when only `forge_unet_storage_dtype` changes, or when all four values stay the same.

`POST /sdapi/v1/unload-checkpoint` does not make a float32 reload use the fp16 weight budget.
