"""Settings and startup hooks for the Forge Turing FP16 extension."""

from modules import errors, script_callbacks, shared

import fp16_switch


def _add_options() -> None:
    section = ("optimizations", "Optimizations")
    category_id = "sd"
    shared.opts.add_option(
        "forge_compute_fp16",
        shared.OptionInfo(
            False,
            "Diffusion compute in fp16",
            section=section,
            category_id=category_id,
        )
        .info("run the loaded UNet in fp16 ; applied once to the current model ; the next generation with this off reloads the original model")
        .info("Turing optimization ; later GPUs may see no speedup and can use more VRAM"),
    )
    shared.opts.add_option(
        "forge_sage_fp16_cast",
        shared.OptionInfo(
            False,
            "Cast attention q/k/v to fp16 for SageAttention",
            section=section,
            category_id=category_id,
        )
        .info("cast non-fp16 attention inputs to fp16 and cast the output back ; stays on the loaded model until a later generation with this off reloads it")
        .info("Turing optimization ; later GPUs may see no speedup and can use more VRAM"),
    )


def on_ui_settings():
    _add_options()


def on_model_loaded(sd_model):
    fp16_switch.reconcile_fp16_switch()


try:
    _add_options()
    fp16_switch.install_patches()
except Exception:
    errors.report("forge-turing-fp16 failed to install", exc_info=True)

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_model_loaded(on_model_loaded)
